import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Subset

SEED = 42

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2023, 0.1994, 0.2010)

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def get_transforms():
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    return transform_train, transform_test


def make_loaders(
    batch_size: int,
    data_dir: str = "./data",
    train_frac: float = 0.9,
    seed: int = SEED,
    num_workers: int = 2,
    pin_memory: bool = True
):

    transform_train, transform_test = get_transforms()

    full_train_aug = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    full_train_clean = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_test
    )

    train_size = int(train_frac * len(full_train_aug))
    val_size = len(full_train_aug) - train_size

    g = torch.Generator().manual_seed(seed)
    train_split, val_split = random_split(full_train_aug, [train_size, val_size], generator=g)

    train_ds = Subset(full_train_aug, train_split.indices)
    val_ds = Subset(full_train_clean, val_split.indices)

    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader
