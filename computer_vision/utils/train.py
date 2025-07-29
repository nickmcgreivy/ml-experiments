from typing import Callable, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from .data import get_dataloaders
from .models import load_model

Tensor = torch.Tensor

def batch_stats(
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
    accuracy = (preds == y).float().mean().item()
    return loss, accuracy
    
def step(opt, loss):
    loss.backward()
    opt.step()
    opt.zero_grad()

def plot_stats(l, acc, model, epoch, i, dl_len, train, id=None):
    model.plot('loss', l, epoch, i, dl_len, train=train, id=id)
    model.plot('accuracy', acc, epoch, i, dl_len, train=train, id=id)

def validate(model, dl, epoch, id=None):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            l, acc = batch_stats(model, batch)
            plot_stats(l.item(), acc, model, epoch, i, len(dl), train=False, id=id)

def train_epoch(model, dl, opt, epoch, id=None):
    model.train()
    for i, batch in enumerate(dl):
        #if (i % 200 == 0) and i > 0:
        #    print(f"Batch {i}/{len(dl)}")
        l, acc = batch_stats(model, batch)
        step(opt, l)
        plot_stats(l.item(), acc, model, epoch, i, len(dl), train=True, id=id)

def train_model(model: nn.Module, 
                train_dl: torch.utils.data.DataLoader, 
                val_dl: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                num_epochs: int,
                id = None,
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
    for epoch in range(num_epochs):
        #(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(model, train_dl, optimizer, epoch, id=id)
        validate(model, val_dl, epoch, id=id)
    return model

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
    train_dl, val_dl = get_dataloaders(hp, preprocess)
    # Create the model (initialization applied automatically in model.__init__())
    model = load_model(hp).to(device)
    # Set up the optimizer
    optimizer = hp.optimizer(model.parameters(), lr=hp.lr)
    # Train the model
    return train_model(model, train_dl, val_dl, optimizer, hp.num_epochs, id=hp.id)

def val_stats(model, hp, device='cpu'):
    preprocess = lambda X, y: (X.to(device), y.to(device))
    _, val_dl = get_dataloaders(hp, preprocess)
    total_loss, total_accurate, num = 0.0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            l, acc = batch_stats(model, batch)
            num += len(batch[0])
            total_loss += l.item() * len(batch[0])
            total_accurate += acc * len(batch[0])
    return total_loss / num, total_accurate / num