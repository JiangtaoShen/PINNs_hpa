import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from utility import get_data_2d
from pinn import PINNs
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def burgers_2d(num_layers, num_nodes, activation_func, epochs, grid_size, learning_rate, test_name, plotshow = False, device ='cpu'):

    seed = 42
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

    # PDEs
    def pde(x, y):
        x.requires_grad_(True)
        dy = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True)[0]
        dy_t = dy[:, 1:2]
        dy_x = dy[:, 0:1]
        dy_xx = torch.autograd.grad(dy_x, x, torch.ones_like(dy_x), create_graph=True)[0][:, 0:1]
        return dy_t + y * dy_x - 0.01 * dy_xx

    # def output_transform(x, y):
    #     return -torch.sin(np.pi * x[:, 0:1]) + x[:, 1:2] * (1 - x[:, 0:1] ** 2) * y

    def plot_results(model, points_test, true_test, x, t, test_name):
        model.eval()
        len_x = len(x)
        len_t = len(t)

        with torch.no_grad():
            pred_test = model(points_test).detach().cpu().numpy()

        pred_field = pred_test.reshape(len_t, len_x).T  # 形状为(空间点数, 时间点数)
        true_field = true_test.cpu().numpy().reshape(len_t, len_x).T
        error_field = np.abs(pred_field - true_field)

        X, T = np.meshgrid(x, t, indexing='ij')  # 空间x作为行，时间t作为列

        plt.figure(figsize=(5, 4))
        plt.pcolormesh(X, T, pred_field, cmap='viridis', shading='gouraud')
        plt.title('Predicted Field', fontsize=15)
        plt.colorbar()
        plt.xlabel('x', fontsize=15)
        plt.ylabel('t', fontsize=15)
        plt.tight_layout()
        # plt.savefig(f'predicted_field_{test_name}.png', dpi=300)
        plt.show()

        plt.figure(figsize=(5, 4))
        plt.pcolormesh(X, T, true_field, cmap='viridis', shading='gouraud')
        plt.title('Ground Truth', fontsize=15)
        plt.colorbar()
        plt.xlabel('x', fontsize=15)
        plt.ylabel('t', fontsize=15)
        plt.tight_layout()
        # plt.savefig(f'true_field_{test_name}.png', dpi=300)
        plt.show()

        plt.figure(figsize=(5, 4))
        plt.pcolormesh(X, T, error_field, cmap='hot_r', shading='gouraud')
        plt.title('Absolute Error', fontsize=15)
        plt.colorbar()
        plt.xlabel('x', fontsize=15)
        plt.ylabel('t', fontsize=15)
        plt.tight_layout()
        # plt.savefig(f'error_field_{test_name}.png', dpi=300)
        plt.show()

    # Loss function
    def losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0):
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
        l1_ab_metric = torch.mean(torch.abs(pred_test - true_test))
        l1_re_metric = torch.mean(torch.abs(pred_test - true_test)) / torch.mean(torch.abs(true_test))
        l2_ab_metric = torch.mean((pred_test - true_test) ** 2)
        l2_re_metric = torch.sqrt(torch.mean((pred_test - true_test) ** 2)) / torch.sqrt(torch.mean(true_test ** 2))

        total_loss = lam_pde * pde_loss + lam_bc * bc_loss + lam_ic * ic_loss

        return total_loss, {
            'pde_loss': pde_loss.item(),
            'bc_loss': bc_loss.item(),
            'ic_loss': ic_loss.item(),
            'total_loss': total_loss.item(),
            'l1_ab_metric': l1_ab_metric.item(),
            'l1_re_metric': l1_re_metric.item(),
            'l2_ab_metric': l2_ab_metric.item(),
            'l2_re_metric': l2_re_metric.item()
        }

    # Initialization
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    try:
        start_time = time.time()

        # Generate training and test points
        num_x = grid_size
        num_y = grid_size
        range_x = torch.tensor([[-1., 1], [0., 0.99]]).to(device)

        points, b1, b2, b3, b4 = get_data_2d(num_x, num_y, range_x, device)

        # Create model with specified parameters
        model = PINNs(in_dim=2, hidden_dim=num_nodes, out_dim=1, num_layer=num_layers, activation=activation_func).to(device)
        # model.apply_output_transform(output_transform)
        model.apply(init_weights)

        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # optimizer = torch.optim.LBFGS(model.parameters(), line_search_fn='strong_wolfe')

        # Training loop
        train_losses_history = []

        for epoch in tqdm(range(epochs), desc=f"Training {test_name}"):
            def closure():
                total_loss, train_loss_dict = losses(model, points, b1, b2, b3, b4, points_test, lam_pde=1.0, lam_bc=1.0, lam_ic=1.0)
                optimizer.zero_grad()
                total_loss.backward()
                if not hasattr(closure, 'latest_loss_dict'):
                    closure.latest_loss_dict = train_loss_dict
                else:
                    closure.latest_loss_dict.update(train_loss_dict)
                return total_loss

            optimizer.step(closure)

            if hasattr(closure, 'latest_loss_dict'):
                train_losses_history.append(closure.latest_loss_dict.copy())

        end_time = time.time()
        runtime = end_time - start_time
        if plotshow == True:
            plot_results(model, points_test, true_test, x, t, test_name)

        # Get final L2 error
        final_l1_ab_error = train_losses_history[-1]['l1_ab_metric']
        final_l1_re_error = train_losses_history[-1]['l1_re_metric']
        final_l2_ab_error = train_losses_history[-1]['l2_ab_metric']
        final_l2_re_error = train_losses_history[-1]['l2_re_metric']
        final_pde_loss = train_losses_history[-1]['pde_loss']
        final_bc_loss = train_losses_history[-1]['bc_loss']
        final_ic_loss = train_losses_history[-1]['ic_loss']
        final_total_loss = train_losses_history[-1]['total_loss']


        print(f"\n{test_name} completed:")
        print(f"Runtime: {runtime:.2f} seconds")
        print(f"Final L2 error: {final_l2_ab_error:.4e}")
        print(f"Configuration: layers={num_layers}, nodes={num_nodes}, activation={activation_func}, epochs={epochs}, sample_size={grid_size}, lr={learning_rate}")

        return {
            'test_name': test_name,
            'final_l1_ab_error': final_l1_ab_error,
            'final_l1_re_error': final_l1_re_error,
            'final_l2_ab_error': final_l2_ab_error,
            'final_l2_re_error': final_l2_re_error,
            'final_pde_loss': final_pde_loss,
            'final_bc_loss': final_bc_loss,
            'final_ic_loss': final_ic_loss,
            'final_total_loss': final_total_loss,
            'runtime': runtime,
            'num_layers': num_layers,
            'num_nodes': num_nodes,
            'activation_func': activation_func,
            'epochs': epochs,
            'grid_size': grid_size,
            'learning_rate': learning_rate
        }

    except Exception as e:
        print(f"Error in {test_name}: {str(e)}")
        return {
            'test_name': test_name,
            'final_l2_error': float('inf'),
            'runtime': 0,
            'error': str(e)
        }