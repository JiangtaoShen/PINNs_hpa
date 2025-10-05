import torch
import matplotlib.pyplot as plt
import numpy as np
import os


def get_data_2d(num_x, num_y, range_x, device):

    # 生成内部网格点
    x = torch.linspace(range_x[0, 0], range_x[0, 1], num_x)
    y = torch.linspace(range_x[1, 0], range_x[1, 1], num_y)
    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
    points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    # 生成四个边界点
    y_b1 = torch.linspace(range_x[1, 0], range_x[1, 1], num_y)
    b1 = torch.stack([torch.full_like(y_b1, range_x[0, 0]), y_b1], dim=1)

    y_b2 = torch.linspace(range_x[1, 0], range_x[1, 1], num_y)
    b2 = torch.stack([torch.full_like(y_b2, range_x[0, 1]), y_b2], dim=1)

    x_b3 = torch.linspace(range_x[0, 0], range_x[0, 1], num_x)
    b3 = torch.stack([x_b3, torch.full_like(x_b3, range_x[1, 0])], dim=1)

    x_b4 = torch.linspace(range_x[0, 0], range_x[0, 1], num_x)
    b4 = torch.stack([x_b4, torch.full_like(x_b4, range_x[1, 1])], dim=1)

    points = points.to(device)
    b1 = b1.to(device)
    b2 = b2.to(device)
    b3 = b3.to(device)
    b4 = b4.to(device)

    return points, b1, b2, b3, b4


def get_data_3d(num_x, num_y, num_z, range_x, device):
    # Generate 3D grid points
    x = torch.linspace(range_x[0, 0], range_x[0, 1], num_x)
    y = torch.linspace(range_x[1, 0], range_x[1, 1], num_y)
    z = torch.linspace(range_x[2, 0], range_x[2, 1], num_z)

    grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
    points = torch.stack([grid_x.reshape(-1),
                          grid_y.reshape(-1),
                          grid_z.reshape(-1)], dim=1)

    # Generate boundary points (6 faces for 3D cube)
    # Face 1: x = x_min
    y_b1, z_b1 = torch.meshgrid(y, z, indexing='ij')
    b1 = torch.stack([torch.full_like(y_b1.reshape(-1), range_x[0, 0]),
                      y_b1.reshape(-1),
                      z_b1.reshape(-1)], dim=1)

    # Face 2: x = x_max
    y_b2, z_b2 = torch.meshgrid(y, z, indexing='ij')
    b2 = torch.stack([torch.full_like(y_b2.reshape(-1), range_x[0, 1]),
                      y_b2.reshape(-1),
                      z_b2.reshape(-1)], dim=1)

    # Face 3: y = y_min
    x_b3, z_b3 = torch.meshgrid(x, z, indexing='ij')
    b3 = torch.stack([x_b3.reshape(-1),
                      torch.full_like(x_b3.reshape(-1), range_x[1, 0]),
                      z_b3.reshape(-1)], dim=1)

    # Face 4: y = y_max
    x_b4, z_b4 = torch.meshgrid(x, z, indexing='ij')
    b4 = torch.stack([x_b4.reshape(-1),
                      torch.full_like(x_b4.reshape(-1), range_x[1, 1]),
                      z_b4.reshape(-1)], dim=1)

    # Face 5: z = z_min
    x_b5, y_b5 = torch.meshgrid(x, y, indexing='ij')
    b5 = torch.stack([x_b5.reshape(-1),
                      y_b5.reshape(-1),
                      torch.full_like(x_b5.reshape(-1), range_x[2, 0])], dim=1)

    # Face 6: z = z_max
    x_b6, y_b6 = torch.meshgrid(x, y, indexing='ij')
    b6 = torch.stack([x_b6.reshape(-1),
                      y_b6.reshape(-1),
                      torch.full_like(x_b6.reshape(-1), range_x[2, 1])], dim=1)

    # Move all tensors to specified device
    points = points.to(device)
    b1, b2, b3, b4, b5, b6 = [b.to(device) for b in [b1, b2, b3, b4, b5, b6]]

    return points, b1, b2, b3, b4, b5, b6


def plot_func_2d(points_test, model, func, range_x, save_path=None, test_name="result"):

    # Automatically determine grid size from points
    num_points = points_test.shape[0]
    grid_size = int(np.sqrt(num_points))

    if grid_size * grid_size != num_points:
        raise ValueError(f"Number of test points ({num_points}) must be a perfect square for grid plotting")

    # Convert range_x to CPU numpy array
    range_x_np = range_x.cpu().numpy() if hasattr(range_x, 'cpu') else range_x

    # Get predictions and true values
    model.eval()
    with torch.no_grad():
        y_pred = model(points_test).cpu().numpy()
    y_true = func(points_test).cpu().numpy()

    # Reshape for plotting
    x = points_test[:, 0].cpu().numpy().reshape(grid_size, grid_size)
    t = points_test[:, 1].cpu().numpy().reshape(grid_size, grid_size)
    z_true = y_true.reshape(grid_size, grid_size)
    z_pred = y_pred.reshape(grid_size, grid_size)
    z_err = np.abs(z_true - z_pred)

    # Calculate error statistics
    max_error = np.max(z_err)
    mean_error = np.mean(z_err)
    l2_error = np.sqrt(np.mean(z_err ** 2))

    # Plot 1: Predicted physical field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_pred, cmap='viridis', shading='gouraud')
    plt.title('Predicted Field', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{test_name}_predicted.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: True physical field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_true, cmap='viridis', shading='gouraud')
    plt.title('Ground Truth', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{test_name}_ground_truth.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: Error field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_err, cmap='Reds', shading='gouraud')
    plt.title('Absolute Error', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/{test_name}_error.png", dpi=300, bbox_inches='tight')
    plt.show()

