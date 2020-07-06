#!/usr/bin/env python3

import os

import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary

from args import argument_parser
from dataloader import dataloader
from gan import Generator, Discriminator, Gan
from plot import plot_loss
from utils import train
from visualize import visualize


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    parser = argument_parser()
    args = parser.parse_args()

    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    train_dl = dataloader(args.dataset, args.batchsize)

    G = Generator(args.dataset, args.latentdim, args.leakyrelu, args.slope)
    D = Discriminator(args.dataset, args.slope, args.droprate)

    G_optimizer = torch.optim.Adam(G.parameters(), lr=args.lr)
    D_optimizer = torch.optim.Adam(D.parameters(), lr=args.lr)
    G_lr_scheduler = torch.optim.lr_scheduler.StepLR(G_optimizer, 5, 0.1)
    D_lr_scheduler = torch.optim.lr_scheduler.StepLR(D_optimizer, 5, 0.1)

    history = {
        'epoch': np.arange(1, args.epoch),
        'G_train_loss': [],
        'D_train_loss': []
    }

    criterion = nn.BCELoss()

    for epoch in range(args.epoch):
        train(G, D,
              G_optimizer, D_optimizer,
              train_dl,
              args.batchsize,
              args.latentdim,
              args.softlabel,
              epoch,
              args.epoch,
              history,
              device,
              criterion)

        G_lr_scheduler.step()
        D_lr_scheduler.step()

    visualize(G, args.latentdim, args.dataset)
    plot_loss(history)

    if os.path.exists('./model') is False:
        os.makedirs('./model')
    torch.save(G.state_dict(), './model/generator.pkl')
    torch.save(D.state_dict(), './model/discriminator.pkl')


if __name__ == '__main__':
    main()
