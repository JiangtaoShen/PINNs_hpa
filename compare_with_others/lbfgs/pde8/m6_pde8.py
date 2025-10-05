import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import pickle
import time
from utility import get_data_2d, pf_func, init_weights
from kan import KAN
# KAN


seed = 42
device = 'cuda:0'

# ----------------------------
# Configurable Parameters
# ----------------------------
model = KAN(width=[2, 5, 5, 1], grid=5, k=3, save_act=False, auto_save=False).to(device)
model.apply(init_weights)
epochs = 1000
grid_size = 101
learning_rate = 0.001
use_lbfgs = True

# Output
output_pkl = f'./output/m6_pde8.pkl'
output_fig_prefix = f"m6_pde8"

# Data parameters
num_x = grid_size
num_y = grid_size
range_x = torch.tensor([[0., 1.], [0., 1.]]).to(device)

# Loss weights
la_pde = 1.0
la_bc = 500.0
la_ic = 1.0

# Reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# ----------------------------
# PDE definition
# ----------------------------
n = 2
k0 = 2 * np.pi * n
def pde(x, y):
    x.requires_grad_(True)

    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    dy_x = dy[:, 0:1]
    dy_y = dy[:, 1:2]

    dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
    dy_yy = torch.autograd.grad(dy_y, x, torch.ones_like(dy_y), create_graph=True)[0][:, 1:2]
    f = k0 ** 2 * torch.sin(k0 * x[:, 0:1]) * torch.sin(k0 * x[:, 1:2])

    return -dy_xx - dy_yy - k0 ** 2 * y - f

# Analytical solution
def func(x):
    return torch.sin(k0 * x[:, 0:1]) * torch.sin(k0 * x[:, 1:2])

# Loss computation
def losses(model, points, b1, b2, b3, b4, points_test):
    points = points.clone().requires_grad_(True)
    pred_points = model(points)
    pded_residual = pde(points, pred_points)
    pde_loss = torch.mean(pded_residual ** 2)

    pred_b1 = model(b1)
    pred_b2 = model(b2)
    pred_b3 = model(b3)
    pred_b4 = model(b4)
    bc_loss = torch.mean((pred_b1) ** 2) + torch.mean((pred_b2) ** 2) + torch.mean((pred_b3) ** 2) + torch.mean((pred_b4) ** 2)

    ic_loss = torch.tensor(0.0, device=device)

    pred_test = model(points_test)
    true_test = func(points_test)
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
num_test = 100
points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)
points_test, _, _, _, _ = get_data_2d(num_test, num_test, range_x, device)

# ----------------------------
# Optimizer
# ----------------------------
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

pf_func(points_test, model, func, num_test, num_test, range_x, output_fig_prefix, "./output")