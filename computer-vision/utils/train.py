from typing import Callable, Tuple, List
import inspect

import torch
from torch import nn
import torch.nn.functional as F

import utils.data as data
import utils.models as models

Tensor = torch.Tensor

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
    def __init__(self, *, dataset='MNIST', **kwargs):
        input_size, _, num_classes = get_dataset_shape(dataset)
        self.input_size = input_size
        self.num_classes = num_classes
        super().__init__(
            dataset = dataset,
            model_type = 'LogisticRegression',
            **kwargs
        )

class MLPHyperParameters(TrainingHyperParameters):
    def __init__(self, *, dataset='MNIST', hidden_width=64, num_layers=3, **kwargs):
        input_size, _, num_classes = get_dataset_shape(dataset)  
        self.input_size = input_size
        self.hidden_widths = [hidden_width] * num_layers
        self.num_classes = num_classes
        super().__init__(
            dataset = dataset, 
            model_type = 'MLP', 
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
    def __init__(self, *, dataset='MNIST', channel_width=8, num_convs=2, hidden_width=128, **kwargs):
        _, input_channels, num_classes = get_dataset_shape(dataset)      
        self.image_width = get_image_width(dataset)
        self.input_channels = input_channels
        self.hidden_width = hidden_width
        self.num_classes = num_classes
        self.channel_widths = [channel_width * (2**i) for i in range(num_convs)]
        self.pool_size = 2
        self.batch_norm = False
        super().__init__(
            dataset = dataset,
            model_type = 'CNN',
            **kwargs
        )

def compute_batch_stats(
        model: nn.Module, 
        batch: tuple[Tensor, Tensor], 
        loss_fn: Callable = F.cross_entropy,
) -> Tuple[Tensor, float]:
    """
    Compute the loss and accuracy for a batch of data.

    Args:
        model (nn.Module): The model to evaluate.
        Xb (Tensor): Input batch of shape (batch_size, input_size).
        yb (Tensor): Target batch of shape (batch_size,).
        loss_fn (Callable): Loss function to compute the loss.

    Returns:
        loss (Tensor): Computed loss for the batch.
        accuracy (float): Accuracy of the model on the batch.
    """
    X, y = batch
    logits = model(X)
    loss = loss_fn(logits, y)
    preds = torch.argmax(logits, dim=1)
    num_accurate = (preds == y).float().sum().item()
    return loss, num_accurate
    
def train_step(
        model: nn.Module,
        batch: tuple[Tensor, Tensor],
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable = F.cross_entropy,
):
    loss, num_accurate = compute_batch_stats(model, batch, loss_fn)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item(), num_accurate

def train_epoch(
        model: nn.Module,
        data_loader,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable = F.cross_entropy,
):
    losses = []
    total_accurate = 0
    model.train()
    for i, batch in enumerate(data_loader):
        if (i % 200 == 0) and i > 0:
            print(f"Batch {i}/{len(data_loader)}")
        loss, num_accurate = train_step(model, batch, optimizer, loss_fn)
        losses.append(loss)
        total_accurate += num_accurate
    return sum(losses) / len(losses), total_accurate / len(data_loader.dataset)

def compute_dataset_stats(
        model: nn.Module, 
        data_loader,
        loss_fn: Callable = F.cross_entropy,
) -> Tuple[float, float]:
    total_loss = 0.0
    total_accuracy = 0.0

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            loss, num_accurate = compute_batch_stats(model, batch, loss_fn)
            total_loss += loss.item() * len(batch[0])
            total_accuracy += num_accurate

    avg_loss = total_loss / len(data_loader.dataset)
    avg_accuracy = total_accuracy / len(data_loader.dataset)
    return avg_loss, avg_accuracy

def append_stats(
        accuracies: List[float],
        losses: List[float],
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
):  
    model.eval()
    with torch.no_grad():
        loss, accuracy = compute_dataset_stats(model, data_loader)
        accuracies.append(accuracy)
        losses.append(loss)

def train_model(model: nn.Module, 
                train_dl: torch.utils.data.DataLoader, 
                val_dl: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                num_epochs: int
):
    """
    Train the model on the training dataset and validate on the validation dataset.

    Args:
        model (nn.Module): The model to train.
        train_dl (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_dl (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        num_epochs (int): Number of epochs to train.

    Returns:
        model (nn.Module): The trained model.
        train_losses: A list of training losses.
        train_accuracies: A list of training accuracies at the end of each epoch.
        val_losses: A list of validation losses at the end of each epoch.
        val_accuracies: A list of validation accuracies at the end of each epoch.
    """
    train_losses = []
    train_accuracies = []
    val_losses = [] # type: list[float]
    val_accuracies = [] # type: list[float]

    append_stats(val_accuracies, val_losses, model, val_dl)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        # Train for one epoch
        train_loss, train_accuracy = train_epoch(model, train_dl, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Append validation stats
        append_stats(val_accuracies, val_losses, model, val_dl)
    
    return model, train_losses, train_accuracies, val_losses, val_accuracies

def fit(hp, device='cpu'):
    """
    Fit a model based on the provided hyperparameters.
    
    Args:
        hp (TrainingHyperParameters): Hyperparameters for training.
    
    Returns:
        model (nn.Module): The trained model.
        train_losses: A list of training losses.
        train_accuracies: A list of training accuracies at the end of each epoch.
        val_losses: A list of validation losses at the end of each epoch.
        val_accuracies: A list of validation accuracies at the end of each epoch.
    """
    preprocess = lambda X, y: (X.to(device), y.to(device))
    # Load the dataset
    train_dl, val_dl = data.get_dataloaders(hp, preprocess)
    # Create the model (initialization applied automatically in model.__init__())
    model = models.load_model(hp).to(device)
    # Set up the optimizer
    optimizer = hp.optimizer(model.parameters(), lr=hp.lr)
    # Train the model
    return train_model(model, train_dl, val_dl, optimizer, hp.num_epochs)