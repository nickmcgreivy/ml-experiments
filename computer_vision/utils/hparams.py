import inspect
from typing import Callable, Tuple, List

import torch
import torch.nn as nn

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        frame_args = inspect.currentframe().f_back.f_locals
        hparams = {k:v for k, v in frame_args.items() 
                        if k not in set(ignore+['self']) 
                             and not k.startswith('_')}
        for k, v in hparams.items():
            setattr(self, k, v)

class TrainingHyperParameters(HyperParameters):
    """ A class to store hyperparameters for training."""
    def __init__(
            self,
            dataset: str,
            model_type: str,
            id: str,
            *,
            activation: str = 'relu',
            batch_norm: bool = False,
            init_scale: float = 1.0,
            init_fn: Callable = nn.init.xavier_normal_,
            optimizer = torch.optim.Adam,
            lr: float = 1e-3,
            batch_size: int = 64,
            num_epochs: int = 5,
            dataset_size: int = 60000,
            **kwargs,
        ):
        self.save_hyperparameters()

def get_dataset_shape(dataset: str):
    if dataset in ['MNIST', 'FashionMNIST']:
        input_size = 784
        input_channels = 1
        num_classes = 10
    elif dataset == 'CIFAR10':
        input_size = 3072
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return input_size, input_channels, num_classes

class LogisticRegressionHyperParameters(TrainingHyperParameters):
    def __init__(self, *, dataset='MNIST', id=None, **kwargs):
        self.input_size, _, self.num_classes = get_dataset_shape(dataset)
        super().__init__(
            dataset = dataset,
            model_type = 'LogisticRegression',
            id = id,
            **kwargs
        )

class MLPHyperParameters(TrainingHyperParameters):
    def __init__(self, *, dataset='MNIST', id=None, hidden_width=64, num_layers=3, **kwargs):
        self.input_size, _, self.num_classes = get_dataset_shape(dataset)  
        self.hidden_widths = [hidden_width] * num_layers
        super().__init__(
            dataset = dataset,
            model_type = 'MLP',
            id = id,
            **kwargs
        )

def get_image_width(dataset):
    match dataset:
        case 'MNIST' | 'FashionMNIST':
            return 28
        case 'CIFAR10':
            return 32
        case _:
            raise ValueError("Unsupported dataset")

class CNNHyperParameters(TrainingHyperParameters):
    def __init__(self, *, dataset='MNIST', id=None, channel_width=8, num_convs=2, hidden_width=128, **kwargs):
        _, self.input_channels, self.num_classes = get_dataset_shape(dataset)      
        self.image_width = get_image_width(dataset)
        self.hidden_width = hidden_width
        self.channel_widths = [channel_width * (2**i) for i in range(num_convs)]
        self.pool_size = 2
        self.batch_norm = False
        super().__init__(
            dataset = dataset,
            model_type = 'CNN',
            id = id,
            **kwargs
        )