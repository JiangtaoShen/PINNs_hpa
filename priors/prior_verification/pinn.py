import torch
import torch.nn as nn
import math


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