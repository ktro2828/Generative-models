#!/usr/bin/env python3

import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dataset', type=str, default='mnist', help='mnist, fashion_mnist or others')
    parser.add_argument('-b', '--batchsize', type=int, default='128', help='batch size')
    parser.add_argument('--epoch', type=int, default=20, help='epochs')
    parser.add_argument('--latentdim', type=int, default=100, help='latent dimention')
    parser.add_argument('--leakyrelu', type=bool, default='True', help='use leakyrelu or not')
    parser.add_argument('--slope', type=float, default=0.2, help='leakyrelu\'s negative slope')
    parser.add_argument('--droprate', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--lr', type=float, default='0.0002', help='learning rate')
    parser.add_argument('--genoptim', type=str, default='adam', help='adam or rmsprop')
    parser.add_argument('--discoptim', type=str, default='adam', help='adam or rmsprop')
    parser.add_argument('--softlabel', type=bool, default=False, help='use softlabel or not')

    return parser
