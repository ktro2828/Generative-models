#!/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def dataloader(dataset_name, batch_size):
    """Load datasets

    Parameters
    ----------
        dataset_name : str
            dataset's name you want to use
        batch_size : int
            batch size
    """
    if dataset_name == 'mnist':
        train_loader = DataLoader(
            datasets.MNIST('./',
                           train=True,
                           dowanload=True,
                           transform=transforms.Compose([
                               transforms.RandomHorizontialFlip(p=0.5),
                               transforms.RandomAffine(
                                   degree=0.2, scale=(0.8, 1.2)),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   mean=[
                                       0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225]
                               )
                           ])),
            batch_size=batch_size,
            shuffle=True
        )
    elif dataset_name == 'fashion_mnist':
        train_loader = DataLoader(
            datasets.FashionMNIST('./',
                                  train=True,
                                  dowanload=True,
                                  transform=transforms.Compose([
                                      transforms.RandomHorizontialFlip(
                                          p=0.5),
                                      transforms.RandomAffine(
                                          degree=0.2, scale=(0.8, 1.2)),
                                      transforms.ToTensor(),
                                      transforms.Normalize(
                                          mean=[
                                              0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]
                                      )
                                  ])),
            batch_size=batch_size,
            shuffle=True
        )
    else:
        train_loader = DataLoader(
            datasets.CelebA('./',
                            train=True,
                            dowanload=True,
                            transform=transforms.Compose([
                                transforms.RandomHorizontialFlip(p=0.5),
                                transforms.RandomAffine(
                                    degree=0.2, scale=(0.8, 1.2)),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    mean=[
                                        0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]
                                )
                            ])),
            batch_size=batch_size,
            shuffle=True
        )

    return train_loader
