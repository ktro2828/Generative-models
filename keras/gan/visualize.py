#!/usr/bin/env python3

import os

import numpy as np
import matplotlib.pyplot as plt


def visualize(generator, latentdim, dataset):
    num_visualize = 100
    noise = np.random.normal(0, 1, size=[num_visualize, latentdim])

    gen_img = generator.predict(noise)
    (img_h, img_w) = gen_img.shape[1:3]

    dim = (10, 10)
    plt.figure(figsize=(10, 10))
    for i in range(gen_img.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(gen_img[i].reshape(img_h, img_w), interpolation='nearrest')
        plt.axis('off')
    plt.tight_layout()
    plt.close()
    if os.path.exists('./generated_img/') is False:
        os.makedirs('./generated_img/')
    plt.savefig('./generated_img/{}.png'.format(dataset))
    print('==> Saved figure')
