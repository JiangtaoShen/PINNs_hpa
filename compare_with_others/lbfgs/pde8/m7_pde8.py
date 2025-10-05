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
epochs = 1000

# functions of pinnsformer (keep shapes compatible with original script)
def get_data(range_x, range_y, num_x, num_y):
    x = np.linspace(range_x[0], range_x[1], num_x)
    y = np.linspace(range_y[0], range_y[1], num_y)
    mesh_x, mesh_y = np.meshgrid(x, y)
    data = np.concatenate((np.expand_dims(mesh_x, -1), np.expand_dims(mesh_y, -1)), axis=-1)
    b1 = data[:, 0, :]   # bottom edge y=min
    b2 = data[:, -1, :]  # top edge y=max
    b3 = data[0, :, :]   # left edge x=min
    b4 = data[-1, :, :]  # right edge x=max
    points = data.reshape(-1, 2)
    return points, b1, b2, b3, b4

# keep make_time_sequence to preserve expected (N, L, 2) shapes; use num_step=1 (no temporal extension)
def make_time_sequence(src, num_step=1, step=0.0):
    dim = num_step
    src = np.repeat(np.expand_dims(src, axis=1), dim, axis=1)  # (N, L, 2)
    for i in range(num_step):
        src[:, i, -1] += step * i
    return src

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# analytical solution and RHS for Helmholtz
def analytical_u(x, y):
    k = 4.0 * torch.pi
    return torch.sin(k * x) * torch.sin(k * y)

def rhs_function(x, y):
    # RHS = (4π)^2 * sin(4π x) sin(4π y)
    k = 4.0 * torch.pi
    return (k**2) * torch.sin(k * x) * torch.sin(k * y)

# output / domain
num_x = 51
num_y = 51
test_size = 50
range_x = torch.tensor([[0., 1.], [0., 1.]]).to(device)  # [0,1]^2

points, b1, b2, b3, b4 = get_data(range_x[0].tolist(), range_x[1].tolist(), num_x, num_y)
points_test, _, _, _, _ = get_data(range_x[0].tolist(), range_x[1].tolist(), test_size, test_size)

# keep shape consistent: use single "time" step so arrays have shape (N,1,2)
points = make_time_sequence(points, num_step=1, step=0.0)
points_test = make_time_sequence(points_test, num_step=1, step=0.0)
b1 = make_time_sequence(b1, num_step=1, step=0.0)
b2 = make_time_sequence(b2, num_step=1, step=0.0)
b3 = make_time_sequence(b3, num_step=1, step=0.0)
b4 = make_time_sequence(b4, num_step=1, step=0.0)

points = torch.tensor(points, dtype=torch.float32, requires_grad=True).to(device)
points_test = torch.tensor(points_test, dtype=torch.float32, requires_grad=False).to(device)
b1 = torch.tensor(b1, dtype=torch.float32, requires_grad=True).to(device)
b2 = torch.tensor(b2, dtype=torch.float32, requires_grad=True).to(device)
b3 = torch.tensor(b3, dtype=torch.float32, requires_grad=True).to(device)
b4 = torch.tensor(b4, dtype=torch.float32, requires_grad=True).to(device)

# keep names to reduce changes: treat second coordinate as y (was t before)
points_x, points_t = points[:, :, 0:1], points[:, :, 1:2]          # x, y
points_test_x, points_test_t = points_test[:, :, 0:1], points_test[:, :, 1:2]
b1_x, b1_t = b1[:, :, 0:1], b1[:, :, 1:2]
b2_x, b2_t = b2[:, :, 0:1], b2[:, :, 1:2]
b3_x, b3_t = b3[:, :, 0:1], b3[:, :, 1:2]
b4_x, b4_t = b4[:, :, 0:1], b4[:, :, 1:2]

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# model and optimizer (unchanged)
model = PINNsformer(d_out=1, d_hidden=512, d_model=32, N=1, heads=2).to(device)
model.apply(init_weights)
optim = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

train_losses_history = []
start_time = time.time()

for epoch in tqdm(range(epochs)):
    def closure():
        # model inputs: (x, y) as before (model signature unchanged)
        pred_points = model(points_x, points_t)          # shape (N, L, 1)
        pred_b1 = model(b1_x, b1_t)
        pred_b2 = model(b2_x, b2_t)
        pred_b3 = model(b3_x, b3_t)
        pred_b4 = model(b4_x, b4_t)

        # First derivatives
        u_x = torch.autograd.grad(pred_points, points_x, grad_outputs=torch.ones_like(pred_points), retain_graph=True, create_graph=True)[0]
        u_y = torch.autograd.grad(pred_points, points_t, grad_outputs=torch.ones_like(pred_points), retain_graph=True, create_graph=True)[0]

        # Second derivatives
        u_xx = torch.autograd.grad(u_x, points_x, grad_outputs=torch.ones_like(u_x), retain_graph=True, create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, points_t, grad_outputs=torch.ones_like(u_y), retain_graph=True, create_graph=True)[0]

        # PDE residual: -u_xx - u_yy - (4π)^2 * u - RHS = 0
        k = 4.0 * torch.pi
        rhs = (k**2) * torch.sin(k * points_x) * torch.sin(k * points_t)
        # ensure shapes match
        residual = -u_xx - u_yy - (k**2) * pred_points - rhs
        loss_pde = torch.mean(residual ** 2)

        # Dirichlet zero on all boundaries: enforce value ~ 0
        loss_bc = (torch.mean(pred_b1 ** 2) + torch.mean(pred_b2 ** 2) +
                   torch.mean(pred_b3 ** 2) + torch.mean(pred_b4 ** 2))

        # L2 metric on test set vs analytical solution
        # pick the first (and only) time step: [:,0:1,:]
        true_test = torch.sin(k * points_test_x) * torch.sin(k * points_test_t)
        true_test = true_test[:, 0:1, :]
        pred_test = model(points_test_x, points_test_t)[:, 0:1, :]
        l2_metric = torch.norm(pred_test - true_test) / torch.norm(true_test)

        loss_total = loss_pde + 500.0*loss_bc

        train_loss_dict = {
            'pde_loss': loss_pde.item(),
            'bc_loss': loss_bc.item(),
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
print(f" Losses - PDE: {final_loss_dict['pde_loss']:.4e} | "f"BC: {final_loss_dict['bc_loss']:.4e} | "f"Total: {final_loss_dict['total_loss']:.4e} | "f"l2: {final_loss_dict['l2_metric']:.4e}")

with open('./output/m7_pde8.pkl', 'wb') as f:
    pickle.dump({
        "train_losses": train_losses_history,
        "training_time": training_time
    }, f)

with torch.no_grad():
    u_pred = model(points_test_x, points_test_t)[:, 0:1]
    u_pred = u_pred.cpu().detach().numpy()
u_pred = u_pred.reshape(test_size, test_size)

k = 4.0 * torch.pi
u_true = torch.sin(k * points_test_x[:, 0:1, :]) * torch.sin(k * points_test_t[:, 0:1, :])
u_true = u_true.cpu().detach().numpy().reshape(test_size, test_size)

# Plot 1: Predicted field
plt.figure(figsize=(5, 4))
plt.pcolormesh(points_test_x[:, 0:1, :].cpu().detach().numpy().reshape(test_size, test_size),
               points_test_t[:, 0:1, :].cpu().detach().numpy().reshape(test_size, test_size),
               u_pred, cmap='rainbow', shading='gouraud')
plt.title('Predicted', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xlim(range_x[0].cpu().numpy()[0], range_x[0].cpu().numpy()[1])
plt.ylim(range_x[1].cpu().numpy()[0], range_x[1].cpu().numpy()[1])
plt.tight_layout()
plt.savefig('./output/m7_pde8_predicted.pdf')
plt.show()

# Plot 2: True physical field
plt.figure(figsize=(5, 4))
plt.pcolormesh(points_test_x[:, 0:1, :].cpu().detach().numpy().reshape(test_size, test_size),
               points_test_t[:, 0:1, :].cpu().detach().numpy().reshape(test_size, test_size),
               u_true, cmap='rainbow', shading='gouraud')
plt.title('Ground Truth', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xlim(range_x[0].cpu().numpy()[0], range_x[0].cpu().numpy()[1])
plt.ylim(range_x[1].cpu().numpy()[0], range_x[1].cpu().numpy()[1])
plt.tight_layout()
plt.savefig('./output/m7_pde8_ground_truth.pdf')
plt.show()

# Plot 3: Error
plt.figure(figsize=(5, 4))
plt.pcolormesh(points_test_x[:, 0:1, :].cpu().detach().numpy().reshape(test_size, test_size),
               points_test_t[:, 0:1, :].cpu().detach().numpy().reshape(test_size, test_size),
               np.abs(u_true - u_pred), cmap='rainbow', shading='gouraud')
plt.title('Error', fontsize=15)
plt.colorbar()
plt.xlabel('x', fontsize=15)
plt.ylabel('y', fontsize=15)
plt.xlim(range_x[0].cpu().numpy()[0], range_x[0].cpu().numpy()[1])
plt.ylim(range_x[1].cpu().numpy()[0], range_x[1].cpu().numpy()[1])
plt.tight_layout()
plt.savefig('./output/m7_pde8_error.pdf')
plt.show()