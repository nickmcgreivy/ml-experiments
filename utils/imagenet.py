import io
import os
import pickle

import numpy as np
import scipy # type: ignore
import boto3 # type: ignore
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms # type: ignore


def get_meta_path():
    return os.path.join(os.getcwd(), 'meta/ILSVRC2012_devkit_t12/')

def get_imagenet_ids():
    meta = scipy.io.loadmat(os.path.join(get_meta_path(), 'data/meta.mat'))
    # First 1000 synsets
    return [str(syn[0][1][0]) for syn in meta['synsets'][:1000]]

def get_imagenet_val_labels():
    # download validation ground truth labels for each index
    val_gt_path = os.path.join(get_meta_path(), 'data/ILSVRC2012_validation_ground_truth.txt')
    return np.loadtxt(val_gt_path, dtype=int) - 1

class S3ImageNetDataset(Dataset):
    def __init__(self, bucket, prefix, split='train', transform=None):
        super().__init__()
        self.files = []
        self.bucket = bucket
        self.prefix = prefix
        self.split = split
        self.transform = transform
        s3_client = boto3.client('s3')

        if split == 'train':
            self.files = []
            self.labels = []
            imagenet_ids = get_imagenet_ids()
            for i, imagenet_id in enumerate(imagenet_ids):
                paginator = s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=f'{prefix}/train/{imagenet_id}')
                for page in pages:
                    for obj in page.get('Contents', []):
                        if obj['Key'].endswith('JPEG'):
                            self.files.append(obj['Key'])
                            self.labels.append(i)
        elif split == 'val':
            val_labels = get_imagenet_val_labels()
            self.files = [f'{prefix}/val/ILSVRC2012_val_{i+1:08d}.JPEG' for i in range(len(val_labels))]
            self.labels = val_labels.tolist()
        else:
            raise ValueError(f'Incorrect split, given {split} expected train or val')
    
    def _get_s3_client(self):
        if not hasattr(self, 's3_client'):
            self.s3_client = boto3.client('s3')
        return self.s3_client

    def __getitem__(self, index):
        s3_client = self._get_s3_client()
        obj = s3_client.get_object(Bucket=self.bucket, Key=self.files[index])
        img_data = obj['Body'].read()
        img = Image.open(io.BytesIO(img_data)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        label = self.labels[index]
        return img, label     
    
    def __len__(self):
        return len(self.files)
    
    def __getstate__(self):
        state = self.__dict__.copy()
        state['s3_client'] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.s3_client = boto3.client('s3')

def get_train_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_imagenet(bucket, prefix, split='train', transform=None):
    if split == 'train':
        filename = 'imagenet-train.pkl'
        if transform is None: transform = get_train_transform()
    elif split == 'val':
        filename = 'imagenet-val.pkl'
        if transform is None: transform = get_val_transform()
    else:
        raise ValueError(f"Split was {split}, expected train or val")
    if os.path.exists(os.path.join(os.getcwd(), filename)):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    else:
        dataset = S3ImageNetDataset(bucket, prefix, split=split, transform=transform)
        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)
        return dataset