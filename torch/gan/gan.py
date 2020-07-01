#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Generator block

        Parameters
        ----------
        dataset_name : str
            datset's name you use
        latent_dim : int
            latent dimention
        leakyrelu : bool
            use Leaky-ReLU or not, if not, use BatchNorm2d + ReLU
        slope : float
            LeakyReLU's negative slope
    """
    def __init__(self, dataset_name, latent_dim, leakyrelu, slope):
        super(Generator, self).__init__()
        if 'mnist' is dataset_name:
            out_channel = 1
        else:
            out_channel = 3

        self.leakyrelu = leakyrelu
        self.slope = slope

        self.linear = nn.Linear(latent_dim, 256 * 14 * 14)
        if self.leakyrelu is False:
            self.bn1 = nn.BartchNorm1d(256 * 14 * 14)
        self.conv1 = nn.Conv2d(256, 128, kernel_size=(3, 3), padding=1)
        if self.leakyrelu is False:
            self.bn2 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1)
        if self.leakyrelu is False:
            self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channel, kernel_size=(1, 1))

    def forward(self, x):
        h = self.linear(x)
        if self.leakyrelu:
            h = F.leaky_relu_(h, self.slope)
        else:
            h = F.relu(self.bn1(h))
        h = self.conv1(h)
        if self.leakyrelu:
            h = F.leaky_relu_(h, self.slope)
        else:
            h = F.relu(self.bn2(h))
        h = self.conv2(h)
        if self.leakyrelu:
            h = F.leaky_relu_(h, self.slope)
        else:
            h = F.relu(self.bn3(h))
        h = self.conv3(h)

        y = F.sigmoid(h)

        return y

class Discriminator(nn.Module):
    """Discriminator block

        Parameters
        ----------
        dataset_name : str
            dataset's name you use
        slope : float
            LeakyReLU's negative slope
        droprate : float
            dropout rate
    """
    def __init__(self, dataset_name, slope, droprate):
        super(Discriminator, self).__init__()
        if 'mnist' in dataset_name:
            in_channel = 1
        else:
            in_channel = 3
        self.slope = slope
        self.droprate = droprate

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2 = nn.conv2d(64, 128, kernel_size=(3, 3), strides=(2, 2), padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), strides=(2, 2), padding=1)
        self.avg_pool = GlobalAvgPool2d()
        self.fc1 = nn.linear(256*7*7, 1024)
        self.bn1 = nn.BartchNorm1d(1024)
        self.fc2 = nn.linear(1024, 1)

    def forward(self, x):
        h = self.conv1(h)
        h = F.leaky_relu_(h, self.slope)
        h = F.dropout2d(h, self.droprate, inplace=True)
        h = self.conv2(h)
        h = F.leaky_relu_(h, self.slope)
        h = F.dropout2d(h, self.droprate, inplace=True)
        h = self.conv3(h)
        h = F.leaky_relu_(h, self.slope)
        h = F.dropout2d(h, self.droprate, inplace=True)
        h = self.avg_pool(h)
        h = self.fc1(h)
        h = F.leaky_relu_(self.bn1(h), self.slope)
        h = self.fc2(h)

        y = F.sigmoid(h)

        return y

    class GlobalAvgPool2d(nn.Module):
        def __init__(self, device=('cpu')):
            super().__init__()

        def forward(self, x):
            return F.avg_pool2d(x, kernel_size=x.size()[2:].view(01, x.size(1)))

class Gan(nn.Module):
    """Gan block

        this block is only used for evaluation
    """
    def __init__(self, dataset_name, latent_dim, leakyrelu, slope, droprate):
        self.generator = Generator(latent_dim, leakyrelu, slope)
        self.discriminator = Discriminator(dataset_name, slope, droprate)

    def forward(self, x):
        x = self.generator(x)
        y = self.discriminator(x)

        return y
