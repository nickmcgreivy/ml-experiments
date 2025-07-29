import os

import torch
from torch.utils.data import Subset, TensorDataset
from torchvision import transforms, datasets # type: ignore

class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield self.func(*b)
    
    @property
    def dataset(self):
        return self.dl.dataset

def get_root():
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)

    # Traverse up the directory tree to find 'ml-experiments'
    project_root = current_dir
    while os.path.basename(project_root) != 'ml-experiments' and project_root != os.path.dirname(project_root):
        project_root = os.path.dirname(project_root)
    
    # Check if we found the project root
    if os.path.basename(project_root) != 'ml-experiments':
        raise RuntimeError("Could not find 'ml-experiments' in the parent directories.")

    return project_root

def get_tensordataset(name):
    datadir = get_root() + "/computer_vision/datasets"
    match name:
        case "MNIST":
            train_ds = datasets.MNIST(root=datadir, train=True, download=True)
            val_ds = datasets.MNIST(root=datadir, train=False, download=True)
        case "FashionMNIST":
            train_ds = datasets.FashionMNIST(root=datadir, train=True, download=True)
            val_ds = datasets.FashionMNIST(root=datadir, train=False, download=True)
        case "CIFAR10":
            train_ds = datasets.CIFAR10(root=datadir, train=True, download=True)
            train_features = torch.tensor(train_ds.data).permute(0, 3, 1, 2) / 255
            train_targets = torch.tensor(train_ds.targets)
            val_ds = datasets.CIFAR10(root=datadir, train=False, download=True)
            val_features = torch.tensor(val_ds.data).permute(0, 3, 1, 2) / 255
            val_targets = torch.tensor(val_ds.targets)
            return TensorDataset(train_features, train_targets), TensorDataset(val_features, val_targets)
        case _:
            raise ValueError("Unsupported Dataset")

    def extract(ds):
        features = ds.data.numpy().transpose(1,2,0)
        features = transforms.functional.to_tensor(features).unsqueeze(1)
        return features, ds.targets

    train_features, train_targets = extract(train_ds)
    val_features, val_targets = extract(val_ds)
    
    return TensorDataset(train_features, train_targets), TensorDataset(val_features, val_targets)
    
def get_dataloaders(hp, func):
    batch_size, dataset_size = hp.batch_size, hp.dataset_size
    train_dataset, val_dataset = get_tensordataset(hp.dataset)
    if dataset_size < len(train_dataset):
        indices = list(torch.randperm(len(train_dataset))[:dataset_size])
        train_dataset = Subset(train_dataset, indices)
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=2*batch_size)
    train_dl = WrappedDataLoader(train_dl, func)
    val_dl = WrappedDataLoader(val_dl, func)
    return train_dl, val_dl