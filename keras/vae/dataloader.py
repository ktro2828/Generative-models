#!/usr/bin/env python3

from keras.datasets import mnist, fashion_mnist

def mnist_loader(dataset):
    """Load MNIST dataset
    """
    if dataset=='mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
    else:
        (x_train, _), (x_test, _) = fashion_mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype('float32') / 255.
    x_test = x_test.reshape(x_test.shape + (1,))

    dataset = {
        'x_train': x_train,
        'x_test': x_test,
    }

    return dataset
