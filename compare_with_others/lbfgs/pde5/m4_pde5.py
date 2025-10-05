import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import pickle
import time
import matplotlib.pyplot as plt
from utility import FLS, get_data_2d, pf_func, init_weights
# FLS


seed = 42
device = 'cuda:0'
beta = 30
# ----------------------------
# Configurable Parameters
# ----------------------------
model = FLS(in_dim=2, hidden_dim=512, out_dim=1, num_layer=4).to(device)
model.apply(init_weights)
epochs = 1000
grid_size = 101
learning_rate = 0.0000386
use_lbfgs = True

# Output
output_pkl = f'./output/m4_pde5.pkl'
output_fig_prefix = f"m4_pde5"

# Data parameters
num_x = grid_size
num_y = grid_size
range_x = torch.tensor([[-1., 1], [0., 0.99]]).to(device)

# Loss weights
la_pde = 1.0
la_bc = 1.0
la_ic = 1.0

# Reproducibility
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

data = np.load('Burgers.npz')
t, x, usol = data['t'], data['x'], data['usol']
len_x = x.shape[0]
len_t = t.shape[0]
data.close()
points_test = np.column_stack((np.meshgrid(x, t, indexing='xy')[0].ravel(), np.meshgrid(x, t, indexing='xy')[1].ravel()))
true_test = usol.T.ravel()
points_test = torch.from_numpy(points_test).float().to(device)
true_test = torch.from_numpy(true_test).float().to(device).unsqueeze(1)

# ----------------------------
# PDE definition
# ----------------------------
def pde(x, y):
    x.requires_grad_(True)
    dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
    dy_t = dy[:, 1:2]
    dy_x = dy[:, 0:1]
    dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
    return dy_t + y * dy_x - 0.01 * dy_xx

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

    pred_b1 = model(b1)
    pred_b2 = model(b2)
    bc_loss = torch.mean(pred_b1 ** 2) + torch.mean(pred_b2 ** 2)

    pred_b3 = model(b3)
    true_b3 = -torch.sin(np.pi * b3[:, 0:1])
    ic_loss = torch.mean((pred_b3 - true_b3) ** 2)

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