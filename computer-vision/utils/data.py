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

def get_tensordataset(name):
    match name:
        case "MNIST":
            train_ds = datasets.MNIST(root='datasets', train=True, download=True)
            val_ds = datasets.MNIST(root='datasets', train=False, download=True)
        case "FashionMNIST":
            train_ds = datasets.FashionMNIST(root='datasets', train=True, download=True)
            val_ds = datasets.FashionMNIST(root='datasets', train=False, download=True)
        case "CIFAR10":
            train_ds = datasets.CIFAR10(root='datasets', train=True, download=True)
            train_features = torch.tensor(train_ds.data).permute(0, 3, 1, 2) / 255
            train_targets = torch.tensor(train_ds.targets)
            val_ds = datasets.CIFAR10(root='datasets', train=False, download=True)
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