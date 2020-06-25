#!/usr/bin/env python3

import argparse

def argument_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d', '--dataset', type=str, required=True, help='mnist or fsahion-mnist')
    parser.add_argument('--latentdim', type=int, default=2, help='latent dimetion')
    parser.add_argument('--epoch', type=int, default=2, help='epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=16, help='batch size')

    return parser
