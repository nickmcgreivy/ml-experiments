from typing import Callable, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from .data import get_dataloaders, WrappedDataLoader
from .models import load_model

Tensor = torch.Tensor

def batch_stats(
        model: nn.Module, 
        batch: tuple[Tensor, Tensor], 
        loss_fn: Callable,
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
    y = batch[-1]
    logits = model(*batch[:-1])
    loss = loss_fn(logits, y)
    preds = torch.argmax(logits, dim=-1)
    accuracy = (preds == y).float().mean().item()
    return loss, accuracy
    
def step(model, opt, loss, **kwargs):
    loss.backward()
    if kwargs.get('max_grad_norm', None):
        torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                       max_norm=kwargs['max_grad_norm'])
    opt.step()
    opt.zero_grad()

def plot_stats(l, acc, model, epoch, i, dl_len, train, **kwargs):
    if kwargs.get('plot_exp', None):
        l = math.exp(l)
    model.plot('loss', l, epoch, i, dl_len, train=train, id=kwargs['id'])
    model.plot('accuracy', acc, epoch, i, dl_len, train=train, id=kwargs['id'])

def validate(model, dl, loss_fn, epoch, **kwargs):
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dl):
            l, acc = batch_stats(model, batch, loss_fn)
            plot_stats(l.item(), acc, model, epoch, i, len(dl), 
                       train=False, **kwargs)

def train_epoch(model, dl, opt, loss_fn, epoch, **kwargs):
    model.train()
    for i, batch in enumerate(dl):
        l, acc = batch_stats(model, batch, loss_fn)
        step(model, opt, l, **kwargs)
        plot_stats(l.item(), acc, model, epoch, i, len(dl), 
                   train=True, **kwargs)

def train_model(model: nn.Module, 
                train_dl: torch.utils.data.DataLoader, 
                val_dl: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                num_epochs: int,
                loss_fn: Callable = F.cross_entropy,
                **kwargs,
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
        train_epoch(model, train_dl, optimizer, loss_fn, epoch, **kwargs)
        validate(model, val_dl, loss_fn, epoch, **kwargs)
    return model

def val_stats(model, hp, loss_fn=F.cross_entropy, device='cpu'):
    preprocess = lambda X, y: (X.to(device), y.to(device))
    _, val_dl = get_dataloaders(hp, preprocess)
    total_loss, total_accurate = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            l, acc = batch_stats(model, batch, loss_fn)
            total_loss += l.item()
            total_accurate += acc
    return total_loss / len(val_dl), total_accurate / len(val_dl)


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
    opt = hp.optimizer(model.parameters(), lr=hp.lr)
    # Train the model
    return train_model(model, train_dl, val_dl, opt, hp.num_epochs, id=hp.id)

def fit_rnn(train_dl, 
            val_dl, 
            model, 
            opt, 
            loss_fn, 
            num_epochs,
            device='cpu',
            id=None,
            max_grad_norm=1.0,
            plot_exp=True,
        ):
    preprocess = lambda X, y: (X.to(device), y.to(device))
    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)
    model.to(device)
    return train_model(model, train_dl, val_dl, opt, num_epochs, 
                loss_fn=loss_fn, id=id, max_grad_norm=max_grad_norm, 
                plot_exp=plot_exp)



def fit_mt():
    pass