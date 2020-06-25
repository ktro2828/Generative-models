#!/usr/bin/env python3

import os

import keras
import numpy as np
from tqdm import tqdm

from args import argument_parser
from dataloader import dataloader
from gan import Gan
from plot import plot_loss
from visualize import visualize


def main():
    parser = argument_parser()
    args = parser.parse_args()

    dataset = dataloader(args.dataset)
    x_train = dataset['x_train']
    x_test = dataset['x_test']

    history = {
        'epoch': np.arange(1, args.epoch+1),
        'd_loss': [],
        'a_loss': []
    }

    batch_size = args.batchsize
    batch_count = int(x_train.shape[0] / batch_size)

    net = Gan(x_train.shape[1:], args.latentdim, args.leakyrelu,
              args.tilt, args.droprate, args.genoptim, args.discoptim)
    gan = net.build()
    generator = net.generator_builder()
    discriminator = net.discriminator_builder()

    for epoch in range(args.epoch):
        for _ in tqdm(range(batch_count)):
            random_latent_vec = np.random.normal(size=(batch_size, args.latentdim))

            gennerated_img = generator.predict(random_latent_vec)

            real_img = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]
            combined_img = np.concatenate([gennerated_img, real_img])

            labels = np.concatenate([np.zeros((batch_size, 1)),
                                    np.ones((batch_size, 1))])

            if args.softlabel:
                labels += 0.05 * np.random.random(labels.shape)
            else:
                labels *= 0.9

            d_loss = discriminator.train_on_batch(combined_img, labels)

            random_latent_vec = np.random.normal(size=(batch_size, args.latentdim))

            misleading_target = np.ones((batch_size, 1))

            a_loss = gan.train_on_batch(random_latent_vec, misleading_target)

            history['d_loss'].append(d_loss)
            history['a_loss'].append(a_loss)

        print('Discriminator loss: {}'.format(d_loss))
        print('Adversarial loss: {}'.format(a_loss))

    visualize(generator, args.latentdim, args.dataset)
    plot_loss(history)

    if os.path.exists('./model') is False:
        os.makedirs('./model')
    generator.save('./model/generator.h5', include_optimizer=False)
    discriminator.save('./model/discriminator.h5', include_optimizer=False)
    gan.save('./model/gan.h5', include_optimizer=False)
    print('==> Saved model')


if __name__ == '__main__':
    main()
