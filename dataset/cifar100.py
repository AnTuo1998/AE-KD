from __future__ import print_function

import os
import socket
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.distributed import DistributedSampler


"""
mean = {
    'cifar100': (0.5071, 0.4867, 0.4408),
}

std = {
    'cifar100': (0.2675, 0.2565, 0.2761),
}
"""


def get_data_folder():
    """
    return server-dependent path to store the data
    """
    data_folder = './data/'

    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    return data_folder


def get_cifar100_dataloaders(batch_size=128, num_workers=8, distribution=False):
    """
    cifar 100
    """
    seed = 0
    import os
    import random
    import torch
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

    g = torch.Generator()
    g.manual_seed(seed)

    data_folder = get_data_folder()
    print(data_folder)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408),
                             (0.2675, 0.2565, 0.2761)),
    ])

    train_set = datasets.CIFAR100(root=data_folder,
                                  download=True,
                                  train=True,
                                  transform=train_transform)

    if distribution:
        train_sampler = DistributedSampler(train_set)
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=train_sampler)
    else:
        train_loader = DataLoader(train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)

    test_set = datasets.CIFAR100(root=data_folder,
                                 download=True,
                                 train=False,
                                 transform=test_transform)
    if distribution:
        test_sampler = DistributedSampler(test_set)
        test_loader = DataLoader(test_set,
                                 batch_size=int(batch_size/2),
                                 num_workers=int(num_workers/2),
                                 sampler=test_sampler)
    else:
        test_loader = DataLoader(test_set,
                                 batch_size=int(batch_size/2),
                                 shuffle=False,
                                 num_workers=int(num_workers/2))

    return train_loader, test_loader
