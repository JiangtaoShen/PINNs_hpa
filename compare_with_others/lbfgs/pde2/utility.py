import torch
import torch.nn as nn
import math
import os
import matplotlib.pyplot as plt
import numpy as np


class Sin(nn.Module):
    """Custom Sin activation function"""

    def __init__(self):
        super(Sin, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Swish(nn.Module):
    """Custom Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class PINNs(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer, activation='tanh'):
        super(PINNs, self).__init__()

        # Define available activation functions
        self.activations = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'sin': Sin(),
            'swish': Swish()
        }

        # Validate activation function
        if activation.lower() not in self.activations:
            raise ValueError(f"Activation function '{activation}' not supported. "
                             f"Available options: {list(self.activations.keys())}")

        # Get the activation function
        self.activation_func = self.activations[activation.lower()]

        # Output transform function (similar to DeepXDE's apply_output_transform)
        self.output_transform = None

        # Build the network layers
        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(self.activation_func)
            else:
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(self.activation_func)

        # Output layer (no activation)
        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

    def forward(self, x):

        raw_output = self.linear(x)

        if self.output_transform is not None:
            return self.output_transform(x, raw_output)
        else:
            return raw_output

    def apply_output_transform(self, transform_func):

        self.output_transform = transform_func

    def remove_output_transform(self):
        """Remove the output transformation function"""
        self.output_transform = None




# baseline implementation of First Layer Sine (FLS) with output transform
# paper: Learning in Sinusoidal Spaces with Physics-Informed Neural Networks
# link: https://arxiv.org/abs/2109.09338

import torch
import torch.nn as nn


class SinAct(nn.Module):
    """Custom sine activation (used only in the first layer of FLS)."""
    def __init__(self):
        super(SinAct, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class FLS(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(FLS, self).__init__()

        layers = []
        for i in range(num_layer - 1):
            if i == 0:
                # First layer: sine activation
                layers.append(nn.Linear(in_features=in_dim, out_features=hidden_dim))
                layers.append(SinAct())
            else:
                # Hidden layers: tanh
                layers.append(nn.Linear(in_features=hidden_dim, out_features=hidden_dim))
                layers.append(nn.Tanh())

        # Output layer: no activation
        layers.append(nn.Linear(in_features=hidden_dim, out_features=out_dim))

        self.linear = nn.Sequential(*layers)

        # Output transform function (optional, like DeepXDE)
        self.output_transform = None

    def forward(self, x):
        raw_output = self.linear(x)
        if self.output_transform is not None:
            return self.output_transform(x, raw_output)
        else:
            return raw_output

    def apply_output_transform(self, transform_func):
        """Apply a custom output transformation"""
        self.output_transform = transform_func

    def remove_output_transform(self):
        """Remove the output transformation"""
        self.output_transform = None



import torch
import torch.nn as nn
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class QRes_block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(QRes_block, self).__init__()
        self.H1 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.H2 = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x1 = self.H1(x)
        x2 = self.H2(x)
        return self.act(x1 * x2 + x1)


class QRes(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layer):
        super(QRes, self).__init__()
        self.N = num_layer - 1
        self.inlayer = QRes_block(in_dim, hidden_dim)
        self.layers = get_clones(QRes_block(hidden_dim, hidden_dim), num_layer - 1)
        self.outlayer = nn.Linear(in_features=hidden_dim, out_features=out_dim)

        # Output transform (可选)
        self.output_transform = None

    def forward(self, x):
        src = self.inlayer(x)
        for i in range(self.N):
            src = self.layers[i](src)
        raw_output = self.outlayer(src)

        if self.output_transform is not None:
            return self.output_transform(x, raw_output)
        else:
            return raw_output

    def apply_output_transform(self, transform_func):
        """Apply a custom output transformation"""
        self.output_transform = transform_func

    def remove_output_transform(self):
        """Remove the output transformation"""
        self.output_transform = None



import torch
import torch.nn as nn
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class WaveAct(nn.Module):
    def __init__(self):
        super(WaveAct, self).__init__()
        self.w1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.w2 = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, x):
        return self.w1 * torch.sin(x) + self.w2 * torch.cos(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=256):
        super(FeedForward, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(d_model, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_ff),
            WaveAct(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.linear(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(EncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x):
        x2 = self.act1(x)
        x = x + self.attn(x2, x2, x2)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads):
        super(DecoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.ff = FeedForward(d_model)
        self.act1 = WaveAct()
        self.act2 = WaveAct()

    def forward(self, x, e_outputs):
        x2 = self.act1(x)
        x = x + self.attn(x2, e_outputs, e_outputs)[0]
        x2 = self.act2(x)
        x = x + self.ff(x2)
        return x

class Encoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x):
        for i in range(self.N):
            x = self.layers[i](x)
        return self.act(x)

class Decoder(nn.Module):
    def __init__(self, d_model, N, heads):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_model, heads), N)
        self.act = WaveAct()

    def forward(self, x, e_outputs):
        for i in range(self.N):
            x = self.layers[i](x, e_outputs)
        return self.act(x)

class PINNsformer(nn.Module):
    def __init__(self, d_out, d_model, d_hidden, N, heads, num_step=5, step=1e-4):
        super(PINNsformer, self).__init__()

        self.linear_emb = nn.Linear(2, d_model)
        self.encoder = Encoder(d_model, N, heads)
        self.decoder = Decoder(d_model, N, heads)
        self.linear_out = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_hidden),
            WaveAct(),
            nn.Linear(d_hidden, d_out)
        )

        self.num_step = num_step
        self.step = step

        # === 输出转换函数（默认 None）===
        self.output_transform = None

    def make_time_sequence(self, src):
        dim = self.num_step
        src = src.unsqueeze(1).repeat(1, dim, 1)  # (N, num_step, 2)
        for i in range(dim):
            src[:, i, 1] += self.step * i
        return src

    def forward(self, x):
        # 只支持传入 (N, 2)，自动扩展时间序列
        src = self.make_time_sequence(x)  # (N, num_step, 2)
        src_emb = self.linear_emb(src)    # (N, num_step, d_model)
        e_outputs = self.encoder(src_emb)
        d_output = self.decoder(src_emb, e_outputs)
        raw_output = self.linear_out(d_output)  # (N, num_step, d_out)

        # === 如果有输出转换函数，就应用 ===
        if self.output_transform is not None:
            return self.output_transform(x, raw_output[:, 0, :])
        else:
            return raw_output[:, 0, :]

    def apply_output_transform(self, transform_func):
        """绑定输出变换函数"""
        self.output_transform = transform_func

    def remove_output_transform(self):
        """移除输出变换函数"""
        self.output_transform = None




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

def pf_func(points_test, model, func, num_x, num_y, range_x, figure_name, save_path):
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Convert range_x to CPU numpy array first
    range_x_np = range_x.cpu().numpy()

    # Get predictions and true values
    with torch.no_grad():
        y_pred = model(points_test).cpu().numpy()
    y_true = func(points_test).cpu().numpy()

    # Reshape output for plotting
    x = points_test[:, 0].cpu().numpy().reshape(num_x, num_y)
    t = points_test[:, 1].cpu().numpy().reshape(num_x, num_y)
    z_true = y_true.reshape(num_x, num_y)
    z_pred = y_pred.reshape(num_x, num_y)
    z_err = np.abs(z_true - z_pred)

    # Plot 1: Predicted physical field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_pred, cmap='rainbow', shading='gouraud')
    plt.title('Predicted', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{figure_name}_predicted.pdf'))
    # plt.show()

    # Plot 2: True physical field
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_true, cmap='rainbow', shading='gouraud')
    plt.title('Ground Truth', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{figure_name}_ground_truth.pdf'))
    # plt.show()

    # Plot 3: Error
    plt.figure(figsize=(5, 4))
    plt.pcolormesh(x, t, z_err, cmap='rainbow', shading='gouraud')
    plt.title('Error', fontsize=15)
    plt.colorbar()
    plt.xlabel('x', fontsize=15)
    plt.ylabel('t', fontsize=15)
    plt.xlim(range_x_np[0][0], range_x_np[0][1])
    plt.ylim(range_x_np[1][0], range_x_np[1][1])
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'{figure_name}_error.pdf'))
    # plt.show()

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)