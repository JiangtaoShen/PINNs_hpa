import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import pickle
import time
from utility import PINNs, get_data_2d, pf_func, init_weights
# RO-PINNs

seed = 42
device = 'cuda:0'

# ----------------------------
# Configurable Parameters
# ----------------------------
model = PINNs(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4, activation= 'tanh').to(device)
model.apply(init_weights)
epochs = 1000
grid_size = 101
learning_rate = 0.001
use_lbfgs = True

# Output
output_pkl = f'./output/m3_pde8.pkl'
output_fig_prefix = f"m3_pde8"

# Data parameters
num_x = grid_size
num_y = grid_size
range_x = torch.tensor([[0., 1.], [0., 1.]]).to(device)

# Loss weights
la_pde = 1.0
la_bc = 500.0
la_ic = 1.0

# RO-PINN parameters
initial_region = 1e-4
sample_num = 1
past_iterations = 10

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
# RO-PINNs Training
# ----------------------------
points_x = points[:, 0:1]
points_t = points[:, 1:2]

gradient_list_overall = []
gradient_list_temp = []
gradient_variance = 1.0
train_losses_history = []
start_time = time.time()

for epoch in tqdm(range(epochs)):
    def closure():
        x_region_sample_list = []
        t_region_sample_list = []

        for _ in range(sample_num):
            scale = np.clip(initial_region / gradient_variance, a_min=0, a_max=0.01)
            x_region_sample = points_x + (torch.rand(points_x.shape).to(points_x.device) * scale)
            t_region_sample = points_t + (torch.rand(points_t.shape).to(points_t.device) * scale)

            x_region_sample_list.append(x_region_sample)
            t_region_sample_list.append(t_region_sample)

        x_region_sample = torch.cat(x_region_sample_list, dim=0)
        t_region_sample = torch.cat(t_region_sample_list, dim=0)

        region_points = torch.cat((x_region_sample, t_region_sample), dim=1)

        total_loss, train_loss_dict = losses(model, region_points, b1, b2, b3, b4, points_test)

        optimizer.zero_grad()
        total_loss.backward()

        grad_vec = []
        for p in model.parameters():
            if p.grad is not None:
                grad_vec.append(p.grad.detach().view(-1).cpu().numpy())
            else:
                grad_vec.append(np.zeros(1))
        grad_vec = np.concatenate(grad_vec)
        gradient_list_temp.append(grad_vec)

        if not hasattr(closure, 'latest_loss_dict'):
            closure.latest_loss_dict = train_loss_dict
        else:
            closure.latest_loss_dict.update(train_loss_dict)

        return total_loss

    optimizer.step(closure)

    if hasattr(closure, 'latest_loss_dict'):
        train_losses_history.append(closure.latest_loss_dict.copy())

    if len(gradient_list_temp) > 0:
        gradient_list_overall.append(np.mean(np.array(gradient_list_temp), axis=0))
        gradient_list_overall = gradient_list_overall[-past_iterations:]
        gradient_array = np.array(gradient_list_overall)
        gradient_variance = (np.std(gradient_array, axis=0) / (np.mean(np.abs(gradient_array), axis=0) + 1e-6)).mean()
        gradient_list_temp = []

    if gradient_variance == 0:
        gradient_variance = 1


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