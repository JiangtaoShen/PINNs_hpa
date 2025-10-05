import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import copy
import random
from torch.optim import LBFGS
from tqdm import tqdm
import pickle
import time
from pinnsformer import PINNsformer

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = 'cuda:0'

learning_rate = 0.001
epochs = 75805

# functions of pinnsformer
def get_data(range_x, range_y, num_x, num_y):
    x = np.linspace(range_x[0], range_x[1], num_x)
    t = np.linspace(range_y[0], range_y[1], num_y)
    mesh_x, mesh_t = np.meshgrid(x, t)
    data = np.concatenate((np.expand_dims(mesh_x, -1), np.expand_dims(mesh_t, -1)), axis=-1)
    b1 = data[:, 0, :]
    b2 = data[:, -1, :]
    b3 = data[0, :, :]
    b4 = data[-1, :, :]
    points = data.reshape(-1, 2)
    return points, b1, b2, b3, b4

def make_time_sequence(src, num_step=5, step=1e-4):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def func(x, t):
    return torch.sin(x - beta * t)

# output
num_x = 51
num_y = 51
test_size = 100
range_x = torch.tensor([[0., 2*torch.pi], [0., 1.]]).to(device)
points, b1, b2, b3, b4 = get_data(range_x[0].tolist(), range_x[1].tolist(), num_x, num_y)
points_test, _, _, _, _ = get_data(range_x[0].tolist(), range_x[1].tolist(), test_size, test_size)

points = make_time_sequence(points, num_step=5, step=1e-4)
points_test = make_time_sequence(points_test, num_step=5, step=1e-4)
b1 = make_time_sequence(b1, num_step=5, step=1e-4)
b2 = make_time_sequence(b2, num_step=5, step=1e-4)
b3 = make_time_sequence(b3, num_step=5, step=1e-4)
b4 = make_time_sequence(b4, num_step=5, step=1e-4)

points = torch.tensor(points, dtype=torch.float32, requires_grad=True).to(device)
points_test = torch.tensor(points_test, dtype=torch.float32, requires_grad=False).to(device)
b1 = torch.tensor(b1, dtype=torch.float32, requires_grad=True).to(device)
b2 = torch.tensor(b2, dtype=torch.float32, requires_grad=True).to(device)
b3 = torch.tensor(b3, dtype=torch.float32, requires_grad=True).to(device)
b4 = torch.tensor(b4, dtype=torch.float32, requires_grad=True).to(device)

points_x, points_t = points[:,:,0:1], points[:,:,1:2]
points_test_x, points_test_t = points_test[:,:,0:1], points_test[:,:,1:2]
b1_x, b1_t = b1[:,:,0:1], b1[:,:,1:2]
b2_x, b2_t = b2[:,:,0:1], b2[:,:,1:2]
b3_x, b3_t = b3[:,:,0:1], b3[:,:,1:2]
b4_x, b4_t = b4[:,:,0:1], b4[:,:,1:2]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# training
model = PINNsformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)
model.apply(init_weights)
# optim = LBFGS(model.parameters(), line_search_fn='strong_wolfe')
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
beta = 30
train_losses_history = []

start_time = time.time()

for epoch in tqdm(range(epochs)):
    def closure():
        pred_points = model(points_x, points_t)
        pred_test = model(points_test_x, points_test_t)
        pred_b1 = model(b1_x, b1_t)
        pred_b2 = model(b2_x, b2_t)
        pred_b3 = model(b3_x, b3_t)
        pred_b4 = model(b4_x, b4_t)

        u_x = torch.autograd.grad(pred_points, points_x, grad_outputs=torch.ones_like(pred_points), retain_graph=True, create_graph=True)[0]
        u_t = torch.autograd.grad(pred_points, points_t, grad_outputs=torch.ones_like(pred_points), retain_graph=True, create_graph=True)[0]

        loss_pde = torch.mean((u_t + beta*u_x) ** 2)
        loss_bc = torch.mean((pred_b1 - pred_b2) ** 2)
        true_b3 = torch.sin(b3_x)
        loss_ic = torch.mean((true_b3 - pred_b3) ** 2)
        true_test = func(points_test_x[:,0:1,:], points_test_t[:,0:1,:])
        l2_metric = torch.norm(pred_test[:,0:1,:] - true_test) / torch.norm(true_test)

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
        if not hasattr(closure, 'latest_loss_dict'):
            closure.latest_loss_dict = train_loss_dict
        else:
            closure.latest_loss_dict.update(train_loss_dict)
        return loss_total
    optim.step(closure)
    if hasattr(closure, 'latest_loss_dict'):
        train_losses_history.append(closure.latest_loss_dict.copy())

end_time = time.time()
training_time = end_time - start_time

final_loss_dict = train_losses_history[-1]
print(f" Losses - PDE: {final_loss_dict['pde_loss']:.4e} | "f"BC: {final_loss_dict['bc_loss']:.4e} | "f"IC: {final_loss_dict['ic_loss']:.4e} | "f"Total: {final_loss_dict['total_loss']:.4e} | "f"l2: {final_loss_dict['l2_metric']:.4e}")

with open('./output/m7_pde1.pkl', 'wb') as f:
    pickle.dump({
        "train_losses": train_losses_history,
        "training_time": training_time
    }, f)

with torch.no_grad():
    u_pred = model(points_test_x, points_test_t)[:,0:1]
    u_pred = u_pred.cpu().detach().numpy()
u_pred = u_pred.reshape(test_size, test_size)

u_true = func(points_test_x[:,0:1,:], points_test_t[:,0:1,:])
u_true = u_true.cpu().detach().numpy()
u_true = u_true.reshape(test_size, test_size)

# Plot 1: Predicted field
plt.figure(figsize=(5, 4))
plt.pcolormesh(points_test_x[:,0:1,:].cpu().detach().numpy().reshape(test_size, test_size), points_test_t[:,0:1,:].cpu().detach().numpy().reshape(test_size, test_size), u_pred, cmap='rainbow', shading='gouraud')
plt.title('Predicted', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('t', fontsize=15)
plt.xlim(range_x[0].cpu().numpy()[0], range_x[0].cpu().numpy()[1])
plt.ylim(range_x[1].cpu().numpy()[0], range_x[1].cpu().numpy()[1])
plt.tight_layout()
plt.savefig('./output/m7_pde1_predicted.pdf')
plt.show()

# Plot 2: True physical field
plt.figure(figsize=(5, 4))
plt.pcolormesh(points_test_x[:,0:1,:].cpu().detach().numpy().reshape(test_size, test_size), points_test_t[:,0:1,:].cpu().detach().numpy().reshape(test_size, test_size), u_true, cmap='rainbow', shading='gouraud')
plt.title('Ground Truth', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('t', fontsize=15)
plt.xlim(range_x[0].cpu().numpy()[0], range_x[0].cpu().numpy()[1])
plt.ylim(range_x[1].cpu().numpy()[0], range_x[1].cpu().numpy()[1])
plt.tight_layout()
plt.savefig('./output/m7_pde1_ground_truth.pdf')
plt.show()

# Plot 3: Error
plt.figure(figsize=(5, 4))
plt.pcolormesh(points_test_x[:,0:1,:].cpu().detach().numpy().reshape(test_size, test_size), points_test_t[:,0:1,:].cpu().detach().numpy().reshape(test_size, test_size), np.abs(u_true - u_pred), cmap='rainbow', shading='gouraud')
plt.title('Error', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('t', fontsize=15)
plt.xlim(range_x[0].cpu().numpy()[0], range_x[0].cpu().numpy()[1])
plt.ylim(range_x[1].cpu().numpy()[0], range_x[1].cpu().numpy()[1])
plt.tight_layout()
plt.savefig('./output/m7_pde1_error.pdf')
plt.show()