import json
import os
from time import time

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader

class Residual(nn.Module):
    def __init__(self, in_channels, out_channels, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 
                               kernel_size=3, padding=1, 
                               stride=strides)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.conv3 = nn.Conv2d(in_channels, out_channels, 
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        return F.relu(X + Y) 

def resnet18(num_classes=10, in_channels=1):
    def resnet_block(in_channels, out_channels, num_residuals):
        blk = []
        for i in range(num_residuals):
            if i == 0 and in_channels != out_channels:
                blk.append(Residual(in_channels, out_channels, strides=2))
            else:
                blk.append(Residual(out_channels, out_channels))
        return nn.Sequential(*blk)
    stem = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32), 
        nn.ReLU()
    )
    body = nn.Sequential(
        resnet_block(32, 32, 2),
        resnet_block(32, 64, 2),
        resnet_block(64, 128, 2),
        resnet_block(128, 256, 2)
    )
    head = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(256, num_classes)
    )
    return nn.Sequential(stem, body, head)

def load_fashion_mnist(batch_size=128):
    train_ds = datasets.FashionMNIST(root='data', transform=transforms.ToTensor(), train=True, download=True)
    val_ds = datasets.FashionMNIST(root='data', transform=transforms.ToTensor(), train=False, download=True)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_dl, val_dl

def loss_fn(X, y, model):
    return F.cross_entropy(model(X), y)

def train_step(
        X: torch.Tensor, y: torch.Tensor, 
        model: nn.Module, opt: torch.optim) -> None:
    loss = loss_fn(X, y, model)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss

def setup(rank, world_size):
    dist.init_process_group('nccl', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

def train_model(rank, world_size):
    setup(rank, world_size)

    lr, batch_size = 0.1, 128
    train_dl, _ = load_fashion_mnist(batch_size=batch_size)
    model = resnet18().to(rank)
    model = DDP(model, device_ids=[rank])
    opt = torch.optim.SGD(model.parameters(), lr)

    num_epochs = 10

    if rank == 0:
        loss_history = []
        print(f"Process {rank} will be tracking loss.")

    t1 = time()
    for epoch in range(num_epochs):
        print(epoch)
        for X, y in train_dl:
            X, y = X.to(rank), y.to(rank)
            loss = train_step(X, y, model, opt)
            
            loss_for_reduce = loss.detach().clone()
            dist.all_reduce(loss_for_reduce, dist.ReduceOp.SUM)
            avg_loss = loss_for_reduce.item() / world_size

            if rank == 0:
                loss_history.append(avg_loss)

    t2 = time()
    print(f"{(t2-t1)/num_epochs:.1f} sec/epoch on device {rank}/{world_size}")

    if rank == 0:
        print("Saving loss history to file")
        with open('multi-gpu-ddp-loss-history.json', 'w') as f:
            json.dump(loss_history, f)

    cleanup()

def setup_multiprocessing():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

def main():
    setup_multiprocessing()
    world_size = torch.cuda.device_count()
    print(f"running on {world_size} gpus")
    mp.spawn(train_model, 
             args=(world_size,), 
             nprocs=world_size, 
             join=True)

if __name__ == '__main__':
    main()