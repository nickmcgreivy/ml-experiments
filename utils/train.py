from typing import Callable, Tuple
import math

import torch
from torch import nn
import torch.nn.functional as F

from .data import get_dataloaders_hp, WrappedDataLoader
from .models import load_model
from .timer import Timer
from .plot import Animator

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
    model.plot('loss', l, epoch, i, dl_len, train=train, id=kwargs.get('id', ""))
    model.plot('accuracy', acc, epoch, i, dl_len, train=train, id=kwargs.get('id', ""))

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
    """
    for epoch in range(num_epochs):
        #(f"Epoch {epoch + 1}/{num_epochs}")
        train_epoch(model, train_dl, optimizer, loss_fn, epoch, **kwargs)
        validate(model, val_dl, loss_fn, epoch, **kwargs)
    return model

def val_stats(model, val_dl, loss_fn=F.cross_entropy, device='cpu'):
    total_loss, total_accurate = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for batch in val_dl:
            l, acc = batch_stats(model, batch, loss_fn)
            total_loss += l.item()
            total_accurate += acc
    return total_loss / len(val_dl), total_accurate / len(val_dl)

def val_stats_hp(model, hp, loss_fn=F.cross_entropy, device='cpu'):
    preprocess = lambda *args: tuple(arg.to(device) for arg in args)
    _, val_dl = get_dataloaders_hp(hp, preprocess)
    return val_stats(model, val_dl, loss_fn=loss_fn, device=device)

def fit(hp, device='cpu'):
    """
    Fit a model based on the provided hyperparameters.
    
    Args:
        hp (TrainingHyperParameters): Hyperparameters for training.
    
    Returns:
        model (nn.Module): The trained model.
    """
    preprocess = lambda *args: tuple(arg.to(device) for arg in args)
    # Load the dataset
    train_dl, val_dl = get_dataloaders_hp(hp, preprocess)
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
    preprocess = lambda *args: tuple(arg.to(device) for arg in args)
    train_dl = WrappedDataLoader(train_dl, preprocess)
    val_dl = WrappedDataLoader(val_dl, preprocess)
    model.to(device)
    return train_model(model, train_dl, val_dl, opt, num_epochs, 
                loss_fn=loss_fn, id=id, max_grad_norm=max_grad_norm, 
                plot_exp=plot_exp)

def fit_mt(train_dl, 
            val_dl, 
            model, 
            opt, 
            loss_fn, 
            num_epochs,
            device='cpu',
            id=None,
            max_grad_norm=1.0,
        ):
    return fit_rnn(train_dl, val_dl, model, opt, loss_fn, 
                   num_epochs, device=device, id=id, 
                   max_grad_norm=max_grad_norm, plot_exp=False)


####################################################################
# Optimization methods practice train functions
####################################################################

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_loss(net, data_iter, loss):
    metric = Accumulator(2)  # Sum of losses, no. of examples
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def train_optimization(trainer_fn, states, hyperparams, data_iter,
                       feature_dim, num_epochs=2):
    w = torch.normal(mean=0.0, std=0.01, size=(feature_dim, 1),
                     requires_grad=True)
    b = torch.zeros((1), requires_grad=True)
    net, loss = lambda X: X @ w + b, squared_loss
    animator = Animator(xlabel='epoch', ylabel='loss',
                        xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y).mean()
            l.backward()
            trainer_fn([w, b], states, hyperparams)
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                animator.add(n/X.shape[0]/len(data_iter),
                             (evaluate_loss(net, data_iter, loss),))
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')
    return timer.cumsum(), animator.Y[0]      

def train_optimization_concise(trainer_fn, hyperparams, data_iter, num_epochs=4):
    # Initialization
    net = nn.Sequential(nn.Linear(5, 1))
    def init_weights(module):
        if type(module) == nn.Linear:
            torch.nn.init.normal_(module.weight, std=0.01)
    net.apply(init_weights)

    optimizer = trainer_fn(net.parameters(), **hyperparams)
    loss = nn.MSELoss(reduction='none')
    animator = Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs], ylim=[0.22, 0.35])
    n, timer = 0, Timer()
    for _ in range(num_epochs):
        for X, y in data_iter:
            optimizer.zero_grad()
            out = net(X)
            y = y.reshape(out.shape)
            l = loss(out, y)
            l.mean().backward()
            optimizer.step()
            n += X.shape[0]
            if n % 200 == 0:
                timer.stop()
                curr_epoch = n/(X.shape[0]*len(data_iter))
                # `MSELoss` computes squared error without the 1/2 factor
                curr_loss = evaluate_loss(net, data_iter, loss) / 2
                animator.add(curr_epoch, curr_loss)
                timer.start()
    print(f'loss: {animator.Y[0][-1]:.3f}, {timer.sum()/num_epochs:.3f} sec/epoch')