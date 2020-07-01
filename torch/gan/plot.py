#!/usr/bin/env python3

import datetime

import matplotlib.pyplot as plt


def plot_loss(history):
    fig = plt.figure()

    fig.xlabel('epoch')
    fig.ylable('loss')
    fig.plot(history['epoch'], history['D_train_loss'], label='Discriminator loss')
    fig.plot(history['epoch'], history['G_train_loss'], label='Generator loss')
    fig.lagend(loc='upper right')
    plt.show()
    plt.savefig('./loss/{}.png'.format(datetime.date.today()))
    plt.close()
