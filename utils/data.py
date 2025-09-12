import os
import requests # type: ignore
import hashlib
import zipfile
import tarfile
import re
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import Subset, TensorDataset, DataLoader
from torchvision import transforms, datasets # type: ignore
from utils.vocab import Vocab


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


class Dataset(ABC):
    @abstractmethod
    def download(self):
        pass

    def get_dataloaders(self):
        train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(self.val_ds, batch_size=self.batch_size)
        return WrappedDataLoader(train_dl, self.put_batch), \
               WrappedDataLoader(val_dl, self.put_batch)

    def put_batch(self, batch):
        if self.device is not None:
            return tuple(arr.to(self.device) for arr in batch)
        else:
            return batch


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func
        self.dataset = self.dl.dataset

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            if self.func is not None:
                yield self.func(b)
            else:
                yield b


########################################################################################
# Computer vision datasets
########################################################################################


class CVDataset(Dataset):
    def __init__(self, batch_size, resize=(28, 28), device=None):
        self.batch_size = batch_size
        self.resize = resize
        self.datadir = get_root() + "/computer_vision/datasets"
        self.device = device
        self.train_ds, self.val_ds = self.download()

    def transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize)
        ])


class MNIST(CVDataset):
    def download(self):
        train_ds = datasets.MNIST(root=self.datadir, transform=self.transforms(), train=True, download=True)
        val_ds = datasets.MNIST(root=self.datadir, transform=self.transforms(), train=False, download=True)
        return train_ds, val_ds


class FashionMNIST(CVDataset):
    def download(self):
        train_ds = datasets.FashionMNIST(root=self.datadir, transform=self.transforms(), train=True, download=True)
        val_ds = datasets.FashionMNIST(root=self.datadir, transform=self.transforms(), train=False, download=True)
        return train_ds, val_ds


class CIFAR10(CVDataset):
    def __init__(self, batch_size, resize=(32, 32), device=None):
        super().__init__(batch_size, resize, device=device)

    def extract(self, ds):
        features = torch.tensor(ds.data).permute(0, 3, 1, 2) / 255
        features = transforms.functional.resize(features, self.resize)
        targets = torch.tensor(ds.targets)
        return features, targets

    def download(self):
        train_ds = datasets.CIFAR10(root=self.datadir, transform=self.transforms(), train=True, download=True)
        val_ds = datasets.CIFAR10(root=self.datadir, transform=self.transforms(), train=False, download=True)
        return train_ds, val_ds
    
    def transforms(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.resize)
        ])


########################################################################################
# Natural language processing datasets
########################################################################################


class NLPDataset(Dataset):
    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def tokenize(self):
        pass

    def download_url(self, url, folder='../data', sha1_hash=None):
        """Download a file to folder and return the local filepath."""
        if not url.startswith('http'):
            # For back compatability
            raise ValueError("URL must start with http:")
        os.makedirs(folder, exist_ok=True)
        fname = os.path.join(folder, url.split('/')[-1])
        # Check if hit cache
        if os.path.exists(fname) and sha1_hash:
            sha1 = hashlib.sha1()
            with open(fname, 'rb') as f:
                while True:
                    data = f.read(1048576)
                    if not data:
                        break
                    sha1.update(data)
            if sha1.hexdigest() == sha1_hash:
                return fname
        # Download
        print(f'Downloading {fname} from {url}...')
        r = requests.get(url, stream=True, verify=True)
        with open(fname, 'wb') as f:
            f.write(r.content)
        return fname


class TimeMachineDataset(NLPDataset):
    def __init__(self, num_steps, batch_size, num_train=10000, num_val=5000, device=None):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_train = num_train
        self.num_val = num_val
        self.device = device
        self.train_ds, self.val_ds, self.vocab = self.download()

    def download(self):
        raw_text = self.download_raw_text()
        corpus, vocab = self.build(raw_text)
        train_ds, val_ds = self.create_dataset(corpus)
        return train_ds, val_ds, vocab

    def download_raw_text(self):
        url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
        folder = get_root() + "/natural_language_processing/datasets"
        sha1_hash = '090b5e7e70c295757f55df93cb0a180b9691891a'
        filename = self.download_url(url, folder, sha1_hash)
        with open(filename) as f:
            return f.read()
        
    def preprocess(self, raw_text):
        return re.sub('[^A-Za-z]+', ' ', raw_text).lower()
    
    def tokenize(self, text):
        return list(text)
    
    def build(self, raw_text, vocab=None):
        tokens = self.tokenize(self.preprocess(raw_text))
        if vocab is None:
            vocab = Vocab(tokens)
        corpus = vocab[tokens]
        return corpus, vocab
    
    def extract_windows(self, corpus, num_steps):
        return torch.tensor([corpus[i:i+num_steps] for i in range(len(corpus) - num_steps + 1)])

    def create_dataset(self, corpus):
        if self.num_train + self.num_val > len(corpus) - 2 * self.num_steps:
            raise ValueError("Number of datapoints larger than corpus length")
        # prevent data leakage
        divider = self.num_train + self.num_steps
        train_corpus, val_corpus = corpus[0 : divider], corpus[divider - 1 : divider + self.num_val + self.num_steps - 1]
        train_array = self.extract_windows(train_corpus, self.num_steps)
        val_array = self.extract_windows(val_corpus, self.num_steps)

        X_train, Y_train = train_array[:-1], train_array[1:]
        X_val, Y_val = val_array[:-1], val_array[1:]
        return TensorDataset(X_train, Y_train), TensorDataset(X_val, Y_val)

    def get_data(self):
        train_dl, val_dl = self.get_dataloaders()
        return train_dl, val_dl, self.vocab
    

class FraEngMTDataset(NLPDataset):
    def __init__(self, batch_size=4, src_vocab=None, tgt_vocab=None, 
                 num_steps=9, num_train=512, num_val=128, device=None):
        self.batch_size = batch_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.num_steps = num_steps
        self.num_train = num_train
        self.num_val = num_val
        self.device = device
        self.train_ds, self.val_ds, self.src_vocab, self.tgt_vocab = \
                                                        self.download()

    def download(self):
        raw_text = self.download_raw_text()
        src, tgt = tokenize_mt(preprocess_mt(raw_text), max_examples = 
                               self.num_train + self.num_val)
        arrays, src_vocab, tgt_vocab = self.build_arrays(src, tgt)
        train_arrays = (array[:self.num_train] for array in arrays)
        val_arrays = (array[self.num_train:] for array in arrays)
        train_ds = TensorDataset(*train_arrays)
        val_ds = TensorDataset(*val_arrays)
        return train_ds, val_ds, src_vocab, tgt_vocab
    
    def extract_zip(self, filename, folder=None):
        """Extract a zip/tar file into folder.

        Defined in :numref:`sec_utils`"""
        base_dir = os.path.dirname(filename)
        _, ext = os.path.splitext(filename)
        assert ext in ('.zip', '.tar', '.gz'), \
                'Only support zip/tar/gz files'
        if ext == '.zip':
            fp = zipfile.ZipFile(filename, 'r')
        else:
            fp = tarfile.open(filename, 'r')
        if folder is None:
            folder = base_dir
        fp.extractall(folder)
    
    def download_raw_text(self):
        url = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
        folder = get_root() + "/natural_language_processing/datasets"
        sha1_hash = '94646ad1522d915e7b0f9296181140edcf86a4f5'
        filename = self.download_url(url, folder, sha1_hash)
        self.extract_zip(filename)
        with open(folder + '/fra-eng/fra.txt', encoding='utf-8') as f:
            return f.read()
        
    def preprocess(self, raw_text):
        # replace non-breaking space, convert upper to lower
        text = raw_text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        # add space before punctuation
        def needs_space(char, prev_char):
            return char in '!.,?' and prev_char != ' '
        out = [' ' + char if i > 0 and needs_space(char, text[i-1]) else 
                                char for i, char in enumerate(text)]
        return ''.join(out)
    
    def tokenize(self, text, max_examples=None):
        src, tgt = [], []
        for i, line in enumerate(text.split('\n')):
            if max_examples and i >= max_examples:
                break
            parts = line.split('\t')
            if len(parts) == 2:
                eng, fr = parts
                src.append([t for t in f'{eng} <eos>'.split(' ') if t])
                tgt.append([t for t in f'{fr} <eos>'.split(' ') if t])
        return src, tgt
    
    def build_array(self, sentences, vocab, is_tgt=False):
        """Creates array of indices from sentences
        
        Args:
            sentences (list): elements are lists of tokens
            vocab (Vocab): builds vocab if None
            is_tgt (bool): determines whether to append <bos> 
                        to each sentence
        
        Returns:
            array (torch.Tensor): (S, T) indices of tokens in vocab, where
                S it number of sentences, T is number of steps
            vocab (Vocabulary)
            valid_len (torch.Tensor): (num_sentences): length of source sequence (w/out padding)
        """
        # trim sentences or append <pad>, len(sentence) == num_steps for all sentences
        pad_or_trim = lambda seq, t: (seq[:t] if len(seq) > t else seq + \
                                            ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
        # prepend <bos> for all sentences if is_tgt=True
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        # create vocab is vocab is None
        if vocab is None:
            vocab = Vocab(sentences, min_freq=2)
        # create index array from sentences
        array = torch.tensor([vocab[s] for s in sentences])
        # compute valid_len from index array
        valid_len = (array != vocab['<pad>']).sum(dim=1)
        return array, vocab, valid_len

    def build_arrays(self, src, tgt):
        """Creates tensors for training and vocabulary from raw text
        
        Args:
            raw_text (str): machine translation corpus, split by \t and \n
        
        Returns:
            arrays (tuple): (src, tgt, src_valid_len, tgt_label)
                src (torch.Tensor): (S, T) tokens of input language, 
                    no <bos>. S it number of sentences, T is number of steps
                tgt (torch.Tensor): (S, T) tokens of target language, begins w/ <bos>
                src_valid_len (torch.Tensor): (S) number of non-padded tokens in src
                tgt_label (torch.Tensor): (S, T) tokens of target language, no <bos>, shifted by one
            src_vocab (Vocab)
            tgt_vocab (Vocab)
        """
        src_array, src_vocab, src_valid_len = self.build_array(src, self.src_vocab)
        tgt_array, tgt_vocab, _ = self.build_array(tgt, self.tgt_vocab, is_tgt=True)
        arrays = (src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:])
        return arrays, src_vocab, tgt_vocab

    def build(self, src_sentences, tgt_sentences, src_vocab, tgt_vocab):
        assert len(src_sentences) == len(tgt_sentences), \
                                "Sentences much be of same length"
        raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
            src_sentences, tgt_sentences)])
        arrays, _, _ = self.build_arrays(raw_text, src_vocab, tgt_vocab)
        return arrays

    def get_data(self):
        train_dl, val_dl = self.get_dataloaders()
        return train_dl, val_dl, self.src_vocab, self.tgt_vocab

########################################################################################
# Optimization dataset
########################################################################################

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_utils`"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    return torch.utils.data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_nasa_data(batch_size=10, n=1500):
    folder = get_root() + "misc/optimization_algorithms/data"
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/airfoil_self_noise.dat'
    sha1_hash = '76e5be1548fd8222e5074cf0faae75edff8cf93f'
    data = np.genfromtxt(download(url, folder, sha1_hash),
                         dtype=np.float32, delimiter='\t')
    data = torch.from_numpy((data - data.mean(axis=0)) / data.std(axis=0))
    data_iter = load_array((data[:n, :-1], data[:n, -1]),
                               batch_size, is_train=True)
    return data_iter, data.shape[1]-1

########################################################################################
# Older, messier computer vision dataset code
########################################################################################

def get_tensordataset(name):
    def extract(ds):
        features = ds.data.numpy().transpose(1,2,0)
        features = transforms.functional.to_tensor(features).unsqueeze(1)
        return features, ds.targets

    def extract_cifar(ds):
        features = torch.tensor(ds.data).permute(0, 3, 1, 2) / 255
        targets = torch.tensor(ds.targets)
        return features, targets

    datadir = get_root() + "/computer_vision/datasets"
    match name:
        case "MNIST" | "FashionMNIST":
            torch_ds = datasets.MNIST if name == "MNIST" else datasets.FashionMNIST
            train_ds = torch_ds(root=datadir, train=True, download=True)
            val_ds = torch_ds(root=datadir, train=False, download=True)
            train_features, train_targets = extract(train_ds)
            val_features, val_targets = extract(val_ds)
            return TensorDataset(train_features, train_targets), TensorDataset(val_features, val_targets)
        case "CIFAR10":
            train_ds = datasets.CIFAR10(root=datadir, train=True, download=True)
            val_ds = datasets.CIFAR10(root=datadir, train=False, download=True)
            train_features, train_targets = extract_cifar(train_ds)
            val_features, val_targets = extract_cifar(val_ds)
            return TensorDataset(train_features, train_targets), TensorDataset(val_features, val_targets)
        case _:
            raise ValueError("Unsupported Dataset")

def get_dataloaders_hp(hp, func=None):
    return get_dataloaders(hp.batch_size, hp.dataset, dataset_size=hp.dataset_size, func=func)

def get_dataloaders(batch_size, dataset, dataset_size=None, func=None):
    train_dataset, val_dataset = get_tensordataset(dataset)
    if dataset_size and dataset_size < len(train_dataset):
        indices = list(torch.randperm(len(train_dataset))[:dataset_size])
        train_dataset = Subset(train_dataset, indices)
    train_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_dataset, batch_size=batch_size)
    if func is not None:
        train_dl = WrappedDataLoader(train_dl, func)
        val_dl = WrappedDataLoader(val_dl, func)
    return train_dl, val_dl


########################################################################################
# Older, messier natural language processing dataset code
########################################################################################


def download(url, folder='../data', sha1_hash=None):
    """Download a file to folder and return the local filepath."""
    if not url.startswith('http'):
        # For back compatability
        raise ValueError("URL must start with http:")
    os.makedirs(folder, exist_ok=True)
    fname = os.path.join(folder, url.split('/')[-1])
    # Check if hit cache
    if os.path.exists(fname) and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f'Downloading {fname} from {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

def download_time_machine():
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt'
    folder = get_root() + "/natural_language_processing/datasets"
    sha1_hash = '090b5e7e70c295757f55df93cb0a180b9691891a'
    filename = download(url, folder, sha1_hash)
    with open(filename) as f:
        return f.read()

def preprocess_tm(raw_text):
    return re.sub('[^A-Za-z]+', ' ', raw_text).lower()

def tokenize_tm(text):
    return list(text)

def build_tm(raw_text, vocab=None):
    tokens = tokenize_tm(preprocess_tm(raw_text))
    if vocab is None:
        vocab = Vocab(tokens)
    corpus = vocab[tokens]
    return corpus, vocab

def extract_windows_tm(corpus, num_steps):
    return torch.tensor([corpus[i:i+num_steps] for i in range(len(corpus) - num_steps + 1)])

def create_time_machine_dataset(num_steps, num_train=10000, num_val=5000):
    raw_text = download_time_machine()
    corpus, vocab = build_tm(raw_text)
    if num_train + num_val > len(corpus) - 2 * num_steps:
        raise ValueError("Number of datapoints larger than corpus length")
    # prevent data leakage
    divider = num_train + num_steps
    train_corpus, val_corpus = corpus[0 : divider], corpus[divider - 1 : divider + num_val + num_steps - 1]
    train_array = extract_windows_tm(train_corpus, num_steps)
    val_array = extract_windows_tm(val_corpus, num_steps)

    X_train, Y_train = train_array[:-1], train_array[1:]
    X_val, Y_val = val_array[:-1], val_array[1:]
    train_ds = TensorDataset(X_train, Y_train)
    val_ds = TensorDataset(X_val, Y_val)
    return train_ds, val_ds, vocab

def get_data_tm(num_steps, batch_size, num_train=10000, num_val=5000):
    train_ds, val_ds, vocab = create_time_machine_dataset(num_steps=num_steps, num_train=num_train, num_val=num_val)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=2*batch_size, shuffle=False)
    return train_dl, val_dl, vocab

def extract_zip(filename, folder=None):
    """Extract a zip/tar file into folder.

    Defined in :numref:`sec_utils`"""
    base_dir = os.path.dirname(filename)
    _, ext = os.path.splitext(filename)
    assert ext in ('.zip', '.tar', '.gz'), \
            'Only support zip/tar/gz files'
    if ext == '.zip':
        fp = zipfile.ZipFile(filename, 'r')
    else:
        fp = tarfile.open(filename, 'r')
    if folder is None:
        folder = base_dir
    fp.extractall(folder)

def download_mtfraeng():
    url = 'http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip'
    folder = get_root() + "/natural_language_processing/datasets"
    sha1_hash = '94646ad1522d915e7b0f9296181140edcf86a4f5'
    filename = download(url, folder, sha1_hash)
    extract_zip(filename)
    with open(folder + '/fra-eng/fra.txt', encoding='utf-8') as f:
        return f.read()

def preprocess_mt(text):
    # replace non-breaking space, convert upper to lower
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # add space before punctuation
    def needs_space(char, prev_char):
        return char in '!.,?' and prev_char != ' '
    out = [' ' + char if i > 0 and needs_space(char, text[i-1]) else 
                            char for i, char in enumerate(text)]
    return ''.join(out)

def tokenize_mt(text, max_examples=None):
    src, tgt = [], []
    for i, line in enumerate(text.split('\n')):
        if max_examples and i >= max_examples:
            break
        parts = line.split('\t')
        if len(parts) == 2:
            eng, fr = parts
            src.append([t for t in f'{eng} <eos>'.split(' ') if t])
            tgt.append([t for t in f'{fr} <eos>'.split(' ') if t])
    return src, tgt

def build_arrays_mt(raw_text, src_vocab=None, tgt_vocab=None, num_steps=9, num_train=512, num_val=128):
    """Creates tensors for training and vocabulary from raw text
    
    Args:
        raw_text (str): machine translation corpus, split by \t and \n
    
    Returns:
        arrays (tuple): (src, tgt, src_valid_len, tgt_label)
            src (torch.Tensor): (S, T) tokens of input language, 
                no <bos>. S it number of sentences, T is number of steps
            tgt (torch.Tensor): (S, T) tokens of target language, begins w/ <bos>
            src_valid_len (torch.Tensor): (S) number of non-padded tokens in src
            tgt_label (torch.Tensor): (S, T) tokens of target language, no <bos>, shifted by one
        src_vocab (Vocab)
        tgt_vocab (Vocab)
    """
    def build_array(sentences, vocab, is_tgt=False):
        """Creates array of indices from sentences
        
        Args:
            sentences (list): elements are lists of tokens
            vocab (Vocab): builds vocab if None
            is_tgt (bool): determines whether to append <bos> 
                           to each sentence
        
        Returns:
            array (torch.Tensor): (S, T) indices of tokens in vocab, where
                S it number of sentences, T is number of steps
            vocab (Vocabulary)
            valid_len (torch.Tensor): (num_sentences): length of source sequence (w/out padding)
        """
        # trim sentences or append <pad>, len(sentence) == num_steps for all sentences
        pad_or_trim = lambda seq, t: (seq[:t] if len(seq) > t else seq + \
                                             ['<pad>'] * (t - len(seq)))
        sentences = [pad_or_trim(s, num_steps) for s in sentences]
        # prepend <bos> for all sentences if is_tgt=True
        if is_tgt:
            sentences = [['<bos>'] + s for s in sentences]
        # create vocab is vocab is None
        if vocab is None:
            vocab = Vocab(sentences, min_freq=2)
        # create index array from sentences
        array = torch.tensor([vocab[s] for s in sentences])
        # compute valid_len from index array
        valid_len = (array != vocab['<pad>']).sum(dim=1)
        return array, vocab, valid_len
    src, tgt = tokenize_mt(preprocess_mt(raw_text), max_examples = 
                                        num_train + num_val)
    src_array, src_vocab, src_valid_len = build_array(src, src_vocab)
    tgt_array, tgt_vocab, _ = build_array(tgt, tgt_vocab, is_tgt=True)
    arrays = (src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:])
    return arrays, src_vocab, tgt_vocab

def build_mt(src_sentences, tgt_sentences, src_vocab, tgt_vocab):
    assert len(src_sentences) == len(tgt_sentences), \
                            "Sentences much be of same length"
    raw_text = '\n'.join([src + '\t' + tgt for src, tgt in zip(
        src_sentences, tgt_sentences)])
    arrays, _, _ = build_arrays_mt(raw_text, src_vocab, tgt_vocab)
    return arrays

def get_data_mt(batch_size=4, num_steps=10, num_train=512, num_val=128):
    arrays, src_vocab, tgt_vocab = build_arrays_mt(download_mtfraeng(), 
                    num_steps=num_steps, num_train=num_train, num_val=num_val)
    train_arrays = (array[:num_train] for array in arrays)
    val_arrays = (array[num_train:] for array in arrays)
    train_ds = TensorDataset(*train_arrays)
    val_ds = TensorDataset(*val_arrays)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)
    return train_dl, val_dl, src_vocab, tgt_vocab