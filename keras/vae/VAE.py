#!/usr/bin/env python3

import os

import keras
from keras import layers
from keras import backend as K
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from args import argument_parser
from dataloader import mnist_loader


def main():
    parser = argument_parser()
    args = parser.parse_args()

    dataset = mnist_loader(args.dataset)
    x_train = dataset['x_train']
    x_test = dataset['x_test']

    input_shape = x_train.shape[1:]
    input_dim = x_train.shape[1] * x_train.shape[2] * x_train.shape[3]
    print(input_dim)
    latent_dim = args.latentdim

    input_img = keras.Input(shape=input_shape)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(input_img)
    x = layers.Conv2D(64, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)

    shape_before_flatting = K.int_shape(x)

    x = layers.Flatten()(x)
    x = layers.Dense(32, activation='relu')(x)

    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)

    z = layers.Lambda(sampling, arguments={'latent_dim': latent_dim})([z_mean, z_log_var])

    decoder_input = layers.Input(K.int_shape(z)[1:])

    x = layers.Dense(np.prod(shape_before_flatting[1:]), activation='relu')(decoder_input)
    x = layers.Reshape(shape_before_flatting[1:])(x)
    x = layers.Conv2DTranspose(32, 3, padding='same', activation='relu', strides=(2, 2))(x)
    x = layers.Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    decoder = Model(decoder_input, x)
    z_decoded = decoder(z)

    y = CustomVariationalLayer(input_dim, z_mean, z_log_var)([input_img, z_decoded])

    vae = Model(input_img, y)
    vae.compile(optimizer='rmsprop', loss=None)
    vae.summary()

    vae.fit(x=x_train, y=None,
            shuffle=True,
            epochs=args.epoch,
            batch_size=args.batch_size,
            validation_data=(x_test, None))

    show_genarated_img(decoder, args.dataset)

    if os.path.exists('./model/') is False:
        os.makedirs('./model/')
    vae.save('./model/vae.h5', include_optimizer=False)
    print('==> Saved model')

def sampling(args, **kwarg):
    z_mean, z_log_var = args
    latent_dim = kwarg['latent_dim']
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1.)

    return z_mean + K.exp(z_log_var) * epsilon

class CustomVariationalLayer(keras.layers.Layer):
    """define VAE-loss
    """
    def __init__(self, input_dim, z_mean, z_log_var):
        super(CustomVariationalLayer, self).__init__()
        self.input_dim = input_dim
        self.z_mean = z_mean
        self.z_log_var = z_log_var

    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)

        reconst_loss = self.input_dim*keras.metrics.binary_crossentropy(x, z_decoded)

        kl_loss = -0.5*K.sum(
            1 + self.z_log_var - K.square(self.z_mean) - K.exp(2*self.z_log_var), axis=-1)

        return K.mean(reconst_loss + kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)

        return x

def show_genarated_img(decoder, dataset_name):
    grid_shape = 15
    digit_size = 28

    fig = np.zeros((digit_size * grid_shape, digit_size * grid_shape))
    # Transpose the space separating [0.05, 0.95] by 15 grids,
    # using gauss-porcent-point-function's invrese function of cumlative distribution function

    # norm.ppf(0.25)=-0.6745
    # norm.ppf(0.50)=0.0
    # norm.ppf(0.75)~0.6745
    grid_x = norm.ppf(np.linspace(0.05, 0.95, grid_shape))
    grid_y = norm.ppf(np.linspace(0.05, 0.95, grid_shape))

    batch_size = 10
    for i, y_i in enumerate(grid_x):
        for j, x_i in enumerate(grid_y):
            z_sample = np.array([[x_i, y_i]])
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            x_decoded = decoder.predict(z_sample, batch_size=batch_size)

            digits = x_decoded.reshape(batch_size, digit_size, digit_size)
            # if you want to show one of them
            # digit = digits[0]
            # show meaned generated image
            digit = np.mean(digits, axis=0)

            fig[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    plt.imshow(fig, cmap='Greys_r')
    plt.show()
    if os.path.exists('./generated_img/') is False:
        os.makedirs('./generated_img/')
    plt.savefig('./generated_img/{}.png'.format(dataset_name))
    plt.close()
    print('==> Saved figure')


if __name__ == '__main__':
    main()
