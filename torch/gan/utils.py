#!/usr/bin/env python3

import torch
import torch.nn as nn
from tqdm import tqdm


def noise_generator(batch_size, latent_dim, device):
    """Generate noise for generator, size=(batch_size, latent_dim, 1, 1)
    """
    random_latent_vec = torch.randn(
        batch_size, latent_dim).view(-1, latent_dim, 1, 1).to(device)

    return random_latent_vec


def cal_loss(preds, label):
    crietion = nn.BCELoss()

    return crietion(preds, label)


def train(G, D,
          G_optimizer, D_optimizer,
          train_dl,
          latent_dim,
          soft_label,
          epoch,
          epochs,
          history):
    """Training for discriminator and generator

    Parameters
    ----------
        G, D : Generator and Discriminator
        G_optimizer, D_optimizer : optimizer
        train_dl : dataset generated by trainloader
        latent_dim : latent dimention
        soft_label : use softlabel or not (type=bool)
        epoch : the epoch training
        epochs : max epoch
        history : loss history (type=dictionary)
    """

    G.train()
    D.train()
    G_train_loss = 0
    D_train_loss = 0
    train_step = len(train_dl)
    batch_size = train_dl.size()[0]

    for real_img, _ in tqdm(train_dl, desc='Epoch: {}/{}'.format(epoch + 1, epochs)):
        # Discriminator training
        # real image
        real_img = real_img.to(device)
        D_result = D(real_img).squeeze()
        y_real = torch.ones(batch_size)
        if soft_label:
            y_real += 0.05 * torch.rand(y_real.size())
        else:
            y_real *= 0.9
        y_real = y_real.to(device)
        D_real_loss = cal_loss(y_real, D_result)

        # fake image(generated image by generator)
        random_latent_vec = noise_generator(batch_size, latent_dim, device)
        G_result = G(random_latent_vec)
        D_result = D(G_result).squeeze()
        y_fake = torch.zeros(batch_size)
        if soft_label:
            y_fake += 0.05 * torch.rand(y_fake.size())
        else:
            y_fake *= 0.9
        y_fake.to(device)
        D_fake_loss = cal_loss(D_result, y_fake)

        D_loss = D_real_loss + D_fake_loss

        D_train_loss += D_loss.item()
        D_optimizer.zero_grad()
        D_loss.backword()
        D_optimizer.step()

        # Generator training
        random_latent_vec = noise_generator(batch_size, latent_dim, device)
        G_result = G(random_latent_vec)
        D_result = D(G_result).squeeze()
        G_loss = cal_loss(D_result, y_real)

        G_train_loss += G_loss.item()
        G_optimizer.zero_grad()
        G_loss.backword()
        G_optimizer.step()

    G_train_loss /= train_step
    D_train_loss /= train_step
    history['G_train_loss'].append(G_train_loss)
    history['D_train_loss'].append(D_train_loss)
