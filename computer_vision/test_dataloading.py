from pathlib import Path
import requests # type: ignore
import pickle 
import gzip
from timeit import timeit
import time

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets # type: ignore

def load_data(dl, n=10):
    """Loads n batches from dataloader"""
    for i, batch in enumerate(dl):
        if i + 1 >= n:
            break

def time_dl(dl):
    t1 = time.time()
    for batch in dl:
        continue
    t2 = time.time()
    return t2 - t1

def get_tensordataset_dataloader():
    DATA_PATH = Path("./datasets")
    PATH = DATA_PATH / "mnist"
    PATH.mkdir(parents=True, exist_ok=True)
    URL = "https://github.com/pytorch/tutorials/raw/main/_static/"
    FILENAME = "mnist.pkl.gz"
    if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
    with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), _, _) = pickle.load(f, encoding="latin-1")
    x_train, y_train = map(torch.tensor, (x_train, y_train))

    ds = TensorDataset(x_train, y_train)
    return DataLoader(ds, batch_size=64, shuffle=True)

def tensor_dataset_runtime():
    """Prints MNIST dataloading time w/ torch.utils.data.TensorDataset"""
    timing = timeit(stmt="load_data(get_tensordataset_dataloader())", 
                    setup="from __main__ import load_data, get_tensordataset_dataloader", 
                    number=5) / (5*10)
    time_dataset = time_dl(get_tensordataset_dataloader())
    print("TensorDataset data loading")
    print(f"Time per batch: {timing*1000:.1f}ms")
    print(f"Time to load entire dataset (50,000 samples): {time_dataset:.2f}s")

def get_torchvision_dl():
    ds = datasets.MNIST(root='./datasets', train=True, 
                            transform=transforms.ToTensor(), download=True)
    return DataLoader(ds, batch_size=64, shuffle=True)

def torchvision_dataset_runtime():
    """Prints MNIST dataloading time w/ torchvision.datasets"""
    timing=timeit(stmt="load_data(get_torchvision_dl())", 
                setup="from __main__ import load_data, get_torchvision_dl", 
                number=5) / (5*10)
    time_dataset = time_dl(get_torchvision_dl())
    print("torchvision.datasets.MNIST data loading")
    print(f"Time per batch = {timing*1000:.1f}ms")
    print(f"Time to load entire dataset (60,000 samples): {time_dataset:.2f}s")

if __name__ == '__main__':
    tensor_dataset_runtime()
    torchvision_dataset_runtime()