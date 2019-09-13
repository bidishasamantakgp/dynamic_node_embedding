from utils import *
from model_parallel import MPPModel

import tensorflow as tf
import numpy as np
import os
import argparse
import logging
import time
import copy

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m%d %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def add_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--rnn_size', type=int, default=10,
                        help='size of RNN hidden state')
    parser.add_argument('--z_dim', type=int, default=15,
                        help='size of latent space')
    parser.add_argument('--n_c', type=int, default=5, help='number of clusters')
    parser.add_argument('--d_dim', type=int, default=1, help='feature dimension')
    parser.add_argument('--h_dim', type=int, default=20, help='hidden state dimension')
    parser.add_argument('--n', type=int, default=84, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='RNN sequence length')
    parser.add_argument('--start', type=int, default=4273,
                        help='starting association')

    parser.add_argument('--hours', type=int, default=24,
                        help='RNN sequence length')
    
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=100,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--k', type=int, default=5, help='dimension of the hidden space')
    parser.add_argument('--T', type=int, default=590, help='maximum time T in training')
    parser.add_argument('--train_size', type=int, default=1501183,
                        help='the training instances to be loaded')
    parser.add_argument('--test_size', type=int, default=100000,
                        help='the training instances to be loaded')
    parser.add_argument('--alpha', type=float, default=0.5, help='probability of sampling')
    parser.add_argument('--sample', type=bool, default=False, help='sampling or training')
    parser.add_argument('--sbm', type=bool, default=True, help='sbm application')
    parser.add_argument('--data_file', type=str, default='data.pkl',
                        help='the pkl file from the series can be loaded')
    parser.add_argument('--out_dir', type=str, default='output',
                        help='the output dir where the model checkpints will be stored')
    return parser

if __name__ == '__main__':
    parser = add_arguments()
    args = parser.parse_args()
    t1 = time.time()

    #with tf.device('/device:GPU:0'):
    args.sbm = False
    model = MPPModel(args)
    t2 = time.time()
    print("Graph creation time", t2-t1 , " Time")
    t1 = time.time()
    model.initialize()
    t2 = time.time()
    print("Graph initialization time", t2-t1, " Time")
    t1 = time.time()
    data_train, data_test = create_samples(args)
    t2 = time.time()
    print("data loading done", t2-t1, len(data_train), len(data_train[-1]))
    #print(data_train)
    #for i in range(25):
    #x, y = next_batch(args, data_train, 21)
    #    print i, x, y
    #print("Train data", data_train[0])
    
    model.train(args, data_train)
