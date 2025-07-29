from typing import List, Callable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .plot import PlotModule

Tensor = torch.Tensor

def linear_init(m, hp):
    """
    Initialize the weights of  model.

    Args:
        m (nn.Module): The MLP model to initialize.
        scale (float): Scaling factor for the initialization.
    """
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        hp.init_fn(m.weight, gain=hp.init_scale * nn.init.calculate_gain(hp.activation))
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def scale_init(hp):
    def apply_fn(m):
        return linear_init(m, hp)
    return apply_fn

def get_activation(activation: str):
    if activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation == 'tanh':
        return F.tanh
    else:
        raise ValueError(f"Unsupported activation function: {activation}")

class LogisticRegression(PlotModule):
    def __init__(self, hp):
        super().__init__()
        self.linear = nn.Linear(hp.input_size, hp.num_classes)
        self.apply(scale_init(hp))
    
    def forward(self, x: Tensor) -> Tensor:
        """ 
        Forward pass for the logistic regression model. 

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = torch.flatten(x, start_dim=1)
        return self.linear(x)

class MLP(PlotModule):
    def __init__(self, hp):
        super().__init__()
        assert len(hp.hidden_widths) > 0, "At least one hidden layer must be specified."
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(hp.input_size, hp.hidden_widths[0]))
        for i in range(len(hp.hidden_widths) - 1):
            self.fc_layers.append(nn.Linear(hp.hidden_widths[i], hp.hidden_widths[i + 1]))
        self.fc_out = nn.Linear(hp.hidden_widths[-1], hp.num_classes)
        self.activation = get_activation(hp.activation)
        self.apply(scale_init(hp))
    
    def forward(self, x: Tensor) -> Tensor:
        """ 
        Forward pass for the MLP model. 

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        x = torch.flatten(x, start_dim=1)
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        return self.fc_out(x)
    
class CNN(PlotModule):
    def __init__(self, hp):
        super().__init__()
        assert len(hp.channel_widths) > 0, "At least one convolutional layer must be specified."
        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(hp.input_channels, hp.channel_widths[0], 
                                    kernel_size=3, padding=1))
        for i in range(len(hp.channel_widths) - 1):
            self.convs.append(nn.Conv2d(hp.channel_widths[i], 
                                        hp.channel_widths[i + 1], 
                                        kernel_size=3, padding=1))
        self.fc_hidden = nn.Linear(hp.channel_widths[-1] * (hp.image_width // 
                                    (2 ** (len(hp.channel_widths) - 1))) ** 2, 
                                    hp.hidden_width)
        self.fc_out = nn.Linear(hp.hidden_width, hp.num_classes)
        self.pool = nn.MaxPool2d(kernel_size=hp.pool_size)
        self.activation = get_activation(hp.activation)
        self.batch_norm = hp.batch_norm
        self.apply(scale_init(hp))
        
        if self.batch_norm:
            self.batch_norms = nn.ModuleList()
            for i in range(len(hp.channel_widths)):
                self.batch_norms.append(nn.BatchNorm2d(hp.channel_widths[i]))
            self.bn_hidden = nn.BatchNorm1d(hp.hidden_width)

    def forward(self, x: Tensor) -> Tensor:
        """ 
        Forward pass for the CNN model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_channels, height, width).
        Returns:
            Tensor: Output tensor of shape (batch_size, num_classes).
        """
        if self.batch_norm:
            for i, (bn, conv) in enumerate(zip(self.batch_norms, self.convs)):
                x = self.activation(bn(conv(x)))
                if i > 0:
                    x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.activation(self.bn_hidden(self.fc_hidden(x)))
        else:
            for i, conv in enumerate(self.convs):
                x = self.activation(conv(x))
                if i > 0:
                    x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.activation(self.fc_hidden(x))
        return self.fc_out(x)

def load_model(hp):
    if hp.model_type == 'LogisticRegression':
        model = LogisticRegression(hp)
    elif hp.model_type == 'MLP':
        model = MLP(hp)
    elif hp.model_type == 'CNN':
        model = CNN(hp)
    else:
        raise ValueError(f"Unsupported model type: {hp.model_type}")
    return model