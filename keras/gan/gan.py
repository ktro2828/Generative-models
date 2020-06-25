#!/usr/bin/env pyhton3

import keras
from keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop


class Gan(object):
    def __init__(self, image, latent_dim, leaky_relu, tilt, drop_rate, gan_optim, disc_optim):
        (self.img_h, self.img_w, self.channels) = image.shape
        self.latent_dim = latent_dim
        self.leaky_relu = leaky_relu
        self.tilt = tilt
        self.drop_rate = drop_rate
        self.optims = {'adam': Adam(lr=0.0002, beta_1=0.5, clipvalue=1.0, decay=1e-8),
                       'rmsprop': RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)}
        self.init = keras.initializers.RandomNormal(stddev=0.02)
        self.gan_optim = gan_optim
        self.disc_optim = disc_optim

    def generator_builder(self):
        """Build generator
        """
        generator_input = keras.Input(shape=(latent_dim,))

        x = layers.Dense(
            128 * 7 * 7, kernel_initializer=self.init)(generator_input)

        if self.leaky_relu:
            x = layers.LeakyReLU(self.tilt)(x)
        else:
            x = layers.BatchNormalizatuin()(x)
            x = layers.Activation('relu')(x)
        x = layers.Reshape((7, 7, 128))(x)

        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)

        if self.leaky_relu:
            x = layers.LeakyReLU(self.tilt)(x)
        else:
            x = layers.BatchNormalizatuin()(x)
            x = layers.Activation('relu')(x)

        x = layers.Conv2DTranspose(64, 4, strides=2, padding='same')(x)

        if self.leaky_relu:
            x = layers.LeakyReLU(self.tilt)(x)
        else:
            x = layers.BatchNormalizatuin()(x)
            x = layers.Activation('relu')(x)

        x = layers.Conv2D(channels, 5, activation='tanh', padding='same')(x)

        generator = keras.models.Model(generator_input, x)
        generator.summary()

        return generator

    def discriminator_builder(self):
        """Build discriminator
        """
        discriminator_input = layers.Input(
            self.img_h, self.img_w, self.channels)

        x = layers.Conv2D(64, 5, strides=2, padding='same',
                          kernel_initializer=self.init)(discriminator_input)
        x = layers.LeakyReLU(self.tilt)(x)
        x = layers.Dropout(self.drop_rate)(x)
        x = layers.Conv2D(128, 5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(self.tilt)(x)
        x = layers.Dropout(self.drop_rate)(x)
        x = layers.Conv2D(256, 5, strides=2, padding='same')(x)
        x = layers.LeakyReLU(self.tilt)(x)
        x = layers.Dropout(self.drop_rate)(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(1, activation='sigmoid')(x)

        discriminator = keras.models.Model(discriminator_input, x)
        discriminator.summary()

        optimizer = self.optims[self.disc_optim]
        discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

        return discriminator

    def build(self):
        generator = self.generator_builder()
        discriminator = self.discriminator_builder()

        discriminator.trainable = False

        gan_input = keras.Input(shape=(self.latent_dim,))
        gan_output = discriminator(generator(gan_input))
        gan = keras.models.Model(gan_input, gan_output)
        optimizer = self.optims[self.gan_optim]
        gan.compile(optimizer=optimizer, loss='binary_crossentropy')

        return gan
