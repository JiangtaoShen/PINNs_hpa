import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import random
from tqdm import tqdm
import pickle
import time
import scipy.io
from pinnsformer import PINNsformer

# ----------------------------
# 随机种子与设备
# ----------------------------
seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cuda:0'

# ----------------------------
# 超参数
# ----------------------------
learning_rate = 0.001
epochs = 1000


# ----------------------------
# PDE7 参数 (Allen–Cahn)
# ----------------------------
d = 0.001
k = 5.0

# ----------------------------
# 数据准备函数
# ----------------------------
def get_data(range_x, range_y, num_x, num_y):
    x = np.linspace(range_x[0], range_x[1], num_x)
    t = np.linspace(range_y[0], range_y[1], num_y)
    mesh_x, mesh_t = np.meshgrid(x, t)
    data = np.concatenate((np.expand_dims(mesh_x, -1), np.expand_dims(mesh_t, -1)), axis=-1)
    b1 = data[:, 0, :]   # x=-1
    b2 = data[:, -1, :]  # x=1
    b3 = data[0, :, :]   # t=0
    points = data.reshape(-1, 2)
    return points, b1, b2, b3


def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def output_transform(x, y):
    return x[:, 0:1] ** 2 * torch.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y

# ----------------------------
# PDE7 初始条件
# ----------------------------
def initial_condition(x):
    return x**2 * np.cos(np.pi * x)

# ----------------------------
# 测试数据 (真实物理场)
# ----------------------------
mat_data = scipy.io.loadmat('Allen_Cahn.mat')
t_data = mat_data['t'].flatten()
x_data = mat_data['x'].flatten()
u_data = mat_data['u']

# 稀疏采样 (每隔一个点取一次)
t_data_sub = t_data[::2]
x_data_sub = x_data[::2]
u_data_sub = u_data[::2, ::2]

xx, tt = np.meshgrid(x_data_sub, t_data_sub, indexing='xy')
points_test = np.column_stack((xx.ravel(), tt.ravel()))
true_test = u_data_sub.ravel()

# points_test = torch.from_numpy(points_test).float().to(device)
true_test = torch.from_numpy(true_test).float().to(device).unsqueeze(1)

# ----------------------------
# 训练/测试数据
# ----------------------------
num_x = 51
num_y = 51
range_x = torch.tensor([[-1., 1.], [0., 1.]]).to(device)

points, b1, b2, b3 = get_data(range_x[0].tolist(), range_x[1].tolist(), num_x, num_y)

# 时间扩展
points = make_time_sequence(points, num_step=5, step=1e-4)
b1 = make_time_sequence(b1, num_step=5, step=1e-4)
b2 = make_time_sequence(b2, num_step=5, step=1e-4)
b3 = make_time_sequence(b3, num_step=5, step=1e-4)
points_test = make_time_sequence(points_test, num_step=5, step=1e-4)

# 转为tensor
points = torch.tensor(points, dtype=torch.float32, requires_grad=True).to(device)
b1 = torch.tensor(b1, dtype=torch.float32, requires_grad=True).to(device)
b2 = torch.tensor(b2, dtype=torch.float32, requires_grad=True).to(device)
b3 = torch.tensor(b3, dtype=torch.float32, requires_grad=True).to(device)
points_test = torch.tensor(points_test, dtype=torch.float32, requires_grad=True).to(device)

points_x, points_t = points[:, :, 0:1], points[:, :, 1:2]
b1_x, b1_t = b1[:, :, 0:1], b1[:, :, 1:2]
b2_x, b2_t = b2[:, :, 0:1], b2[:, :, 1:2]
b3_x, b3_t = b3[:, :, 0:1], b3[:, :, 1:2]
points_test_x, points_test_t = points_test[:, :, 0:1], points_test[:, :, 1:2]

# ---------------------------
# 初始化模型
# ----------------------------
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

model = PINNsformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)
model.apply(init_weights)
optim = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

train_losses_history = []
start_time = time.time()

# ----------------------------
# 训练循环
# ----------------------------
for epoch in tqdm(range(epochs)):
    def closure():
        pred_points = model(points_x, points_t)

        # PDE7 Allen–Cahn: u_t = d*u_xx + k(u - u^3)
        u_t = torch.autograd.grad(pred_points, points_t,
                                  grad_outputs=torch.ones_like(pred_points),
                                  retain_graph=True, create_graph=True)[0]
        u_x = torch.autograd.grad(pred_points, points_x,
                                  grad_outputs=torch.ones_like(pred_points),
                                  retain_graph=True, create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, points_x,
                                   grad_outputs=torch.ones_like(u_x),
                                   retain_graph=True, create_graph=True)[0]

        loss_pde = torch.mean((u_t - (d * u_xx + k * (pred_points - pred_points**3))) ** 2)

        # 边界条件: u(-1,t) = -1, u(1,t) = -1
        pred_b1 = model(b1_x, b1_t)
        pred_b2 = model(b2_x, b2_t)
        loss_bc = torch.mean((pred_b1 + 1.0) ** 2) + torch.mean((pred_b2 + 1.0) ** 2)

        # 初始条件: u(x,0) = x^2 * cos(pi*x)
        pred_b3 = model(b3_x, b3_t)
        true_b3 = b3_x**2 * torch.cos(np.pi * b3_x)
        loss_ic = torch.mean((pred_b3 - true_b3) ** 2)

        # 测试集 L2 误差
        pred_test = model(points_test_x, points_test_t)[:, 0:1, :]
        pred_test = pred_test.squeeze(-1)
        l2_metric = torch.norm(pred_test - true_test) / torch.norm(true_test)

        loss_total = loss_pde + loss_bc + loss_ic

        train_loss_dict = {
            'pde_loss': loss_pde.item(),
            'bc_loss': loss_bc.item(),
            'ic_loss': loss_ic.item(),
            'total_loss': loss_total.item(),
            'l2_metric': l2_metric.item()
        }

        optim.zero_grad()
        loss_total.backward()
        closure.latest_loss_dict = train_loss_dict
        return loss_total

    optim.step(closure)
    if hasattr(closure, 'latest_loss_dict'):
        train_losses_history.append(closure.latest_loss_dict.copy())

end_time = time.time()
training_time = end_time - start_time

# ----------------------------
# 保存训练结果
# ----------------------------
final_loss_dict = train_losses_history[-1]
print(f" Losses - PDE: {final_loss_dict['pde_loss']:.4e} | "
      f"BC: {final_loss_dict['bc_loss']:.4e} | "
      f"IC: {final_loss_dict['ic_loss']:.4e} | "
      f"Total: {final_loss_dict['total_loss']:.4e} | "
      f"L2: {final_loss_dict['l2_metric']:.4e}")

with open('./output/m7_pde7.pkl', 'wb') as f:
    pickle.dump({
        "train_losses": train_losses_history,
        "training_time": training_time
    }, f)

# ----------------------------
# 可视化预测结果
# ----------------------------
with torch.no_grad():
    u_pred = model(points_test_x, points_test_t)[:,0:1,:]
    u_pred = u_pred.cpu().detach().numpy()

u_pred = u_pred.reshape(len(t_data_sub), len(x_data_sub))
u_true = true_test.cpu().detach().numpy().reshape(len(t_data_sub), len(x_data_sub))

# 预测结果
plt.figure(figsize=(5, 4))
plt.imshow(u_pred, extent=[x_data_sub.min(), x_data_sub.max(), t_data_sub.min(), t_data_sub.max()],
           origin='lower', aspect='auto', cmap='rainbow')
plt.title('Predicted', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('t', fontsize=15)
plt.tight_layout()
plt.savefig('./output/m7_pde7_predicted.pdf')
plt.show()

# 真解
plt.figure(figsize=(5, 4))
plt.imshow(u_true, extent=[x_data_sub.min(), x_data_sub.max(), t_data_sub.min(), t_data_sub.max()],
           origin='lower', aspect='auto', cmap='rainbow')
plt.title('Ground Truth', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('t', fontsize=15)
plt.tight_layout()
plt.savefig('./output/m7_pde7_ground_truth.pdf')
plt.show()

# 误差
plt.figure(figsize=(5, 4))
plt.imshow(np.abs(u_true - u_pred), extent=[x_data_sub.min(), x_data_sub.max(), t_data_sub.min(), t_data_sub.max()],
           origin='lower', aspect='auto', cmap='rainbow')
plt.title('Error', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('t', fontsize=15)
plt.tight_layout()
plt.savefig('./output/m7_pde7_error.pdf')
plt.show()