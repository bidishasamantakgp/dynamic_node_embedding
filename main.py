from utils import *
from model import MPPModel

import tensorflow as tf
import numpy as np
import os
import argparse

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def add_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--rnn_size', type=int, default=3,
                        help='size of RNN hidden state')
    parser.add_argument('--latent_size', type=int, default=3,
                        help='size of latent space')
    parser.add_argument('--batch_size', type=int, default=3000,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=500,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--chunk_samples', type=int, default=1,
                        help='number of samples per mdct chunk')
    return parser

if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    data_train, data_test = create_samples(args)
    model = MPPModel(args)
    model.train(args)