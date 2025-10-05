import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import pickle
import time
import scipy
import scipy.io
import matplotlib.pyplot as plt
from utility import PINNs, get_data_2d, pf_func, init_weights
# PINNs


seed = 42
device = 'cuda:0'

# ----------------------------
# Configurable Parameters
# ----------------------------
model = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4, activation= 'tanh').to(device)
model.apply(init_weights)
epochs = 1000
grid_size = 101
learning_rate = 0.0000386
use_lbfgs = True

# Output
output_pkl = f'./output/m2_pde7.pkl'
output_fig_prefix = f"m2_pde7"

# Data parameters
num_x = grid_size
num_y = grid_size
range_x = torch.tensor([[-1., 1.], [0., 1.]]).to(device)

# Loss weights
la_pde = 1.0
la_bc = 1.0
la_ic = 1.0

# Reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

mat_data = scipy.io.loadmat('Allen_Cahn.mat')
t = mat_data['t'].flatten()
x = mat_data['x'].flatten()
u = mat_data['u']
xx, tt = np.meshgrid(x, t, indexing='xy')
points_test = np.column_stack((xx.ravel(), tt.ravel()))
true_test = u.ravel()
points_test = torch.from_numpy(points_test).float().to(device)
true_test = torch.from_numpy(true_test).float().to(device).unsqueeze(1)

# ----------------------------
# PDE definition
# ----------------------------
d = 0.001
def pde(x, y):
    x.requires_grad_(True)
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    dy_t = dy[:, 1:2]
    dy_x = dy[:, 0:1]
    dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
    return dy_t - d * dy_xx - 5 * (y - y ** 3)

def output_transform(x, y):
    return x[:, 0:1] ** 2 * torch.cos(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y

def plot_results(model, points_test, true_test, x, t):
    model.eval()
    len_x = len(x)
    len_t = len(t)

    with torch.no_grad():
        pred_test = model(points_test).detach().cpu().numpy()

    pred_field = pred_test.reshape(len_t, len_x).T  # 形状为(空间点数, 时间点数)
    true_field = true_test.cpu().numpy().reshape(len_t, len_x).T
    error_field = np.abs(pred_field - true_field)

    X, T = np.meshgrid(x, t, indexing='ij')  # 空间x作为行，时间t作为列

    # Predicted Field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(X, T, pred_field, cmap='rainbow', shading='gouraud')
    plt.title('Predicted', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'./output/{output_fig_prefix}_predicted.pdf', dpi=300)
    plt.show()

    # Ground Truth
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(X, T, true_field, cmap='rainbow', shading='gouraud')
    plt.title('Ground Truth', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'./output/{output_fig_prefix}_ground_truth.pdf', dpi=300)
    plt.show()

    # Absolute Error
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(X, T, error_field, cmap='rainbow', shading='gouraud')
    plt.title('Error', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.tight_layout()
    plt.savefig(f'./output/{output_fig_prefix}_error.pdf', dpi=300)
    plt.show()

# Loss computation
def losses(model, points, b1, b2, b3, b4, points_test):
    points = points.clone().requires_grad_(True)
    pred_points = model(points)
    pde_residual = pde(points, pred_points)
    pde_loss = torch.mean(pde_residual ** 2)

    bc_loss = torch.tensor(0.0, device=device)
    ic_loss = torch.tensor(0.0, device=device)

    pred_test = model(points_test)
    l2_metric = torch.norm(pred_test - true_test) / torch.norm(true_test)

    total_loss = la_pde * pde_loss + la_bc * bc_loss + la_ic * ic_loss
    return total_loss, {
        'pde_loss': pde_loss.item(),
        'bc_loss': bc_loss.item(),
        'ic_loss': ic_loss.item(),
        'total_loss': total_loss.item(),
        'l2_metric': l2_metric.item()
    }

# ----------------------------
# Data Generation
# ----------------------------

points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)

# ----------------------------
# Optimizer
# ----------------------------
model.apply_output_transform(output_transform)
if use_lbfgs:
    optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')
else:
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ----------------------------
# Training
# ----------------------------
train_losses_history = []
start_time = time.time()

for epoch in tqdm(range(epochs)):
    def closure():
        total_loss, train_loss_dict = losses(model, points, b1, b2, b3, b4, points_test)
        optimizer.zero_grad()
        total_loss.backward()
        closure.latest_loss_dict = train_loss_dict
        return total_loss
    optimizer.step(closure)
    train_losses_history.append(closure.latest_loss_dict.copy())

end_time = time.time()
training_time = end_time - start_time
print(f"Training finished in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")

# ----------------------------
# Results & Save
# ----------------------------
final_loss_dict = train_losses_history[-1]
print(
    f"PDE: {final_loss_dict['pde_loss']:.4e} | "
    f"BC: {final_loss_dict['bc_loss']:.4e} | "
    f"IC: {final_loss_dict['ic_loss']:.4e} | "
    f"Total: {final_loss_dict['total_loss']:.4e} | "
    f"L2: {final_loss_dict['l2_metric']:.4e}"
)

with open(output_pkl, 'wb') as f:
    pickle.dump({
        "train_losses": train_losses_history,
        "training_time": training_time
    }, f)

plot_results(model, points_test, true_test, x, t)