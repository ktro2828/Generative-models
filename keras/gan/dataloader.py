#!/usr/bin/env python3

from keras.datasets import mnist, fashion_mnist, cifar10

def dataloader(dataset_name):
    if dataset_name == 'mnist':
        (x_train, _), (x_test, _) = mnist.load_data()
    elif dataset_name == 'fashion_mnist':
        (x_train, _), (x_test, _) = fashion_mnist.load_data()
    else:
        (x_train, _), (x_test, _) = cifar10.load_data(label_mode=fine)

    # Because activation is tabh, each pixel has value in [-1, 1]
    x_train = (x_train.astype('float32')-127.5)/127.5
    x_train = x_train.reshape(x_trian.shape + (1,))
    x_test = (x_test.astype('float32')-127.5)/127.5
    x_test = x_test.reshape(x_test.shape + (1,))

    dataset = {
        'x_train': x_train,
        'x_test': x_test
    }

    return dataset
