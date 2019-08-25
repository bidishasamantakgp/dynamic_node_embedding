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

    parser.add_argument('--rnn_size', type=int, default=3,
                        help='size of RNN hidden state')
    parser.add_argument('--z_dim', type=int, default=3,
                        help='size of latent space')

    parser.add_argument('--n_c', type=int, default=5, help='number of clusters')
    parser.add_argument('--d_dim', type=int, default=1, help='feature dimension')
    parser.add_argument('--h_dim', type=int, default=40, help='hidden state dimension')
    parser.add_argument('--n', type=int, default=84, help='number of nodes')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=100,
                        help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=1.,
                        help='decay of learning rate')
    parser.add_argument('--k', type=int, default=5, help='dimension of the hidden space')
    parser.add_argument('--T', type=int, default=10000, help='maximum time T in training')
    parser.add_argument('--train_size', type=int, default=1720500,
                        help='the training instances to be loaded')
    parser.add_argument('--alpha', type=float, default=0.5, help='probability of sampling')
    parser.add_argument('--sample', type=bool, default=False, help='sampling or training')

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
    model = MPPModel(args)
    #model.initialize()
    t2 = time.time()
    print("Graph creation time", t2-t1 , " Time")
    t1 = time.time()
    model.initialize()
    t2 = time.time()
    print("Graph initialization time", t2-t1, " Time")
    t1 = time.time()
    data_train, data_test = create_samples(args)
    t2 = time.time()
    print("data loading done", t2-t1, len(data_train), data_train[0])

    #x, y = next_batch(args, data_train, 0)
    #print x, y
    '''
    adj_list_prev = []
    adj_list = []
    samples = data_train[:2]
    adj_old = starting_adj(args, samples)
    for b in range(len(samples)):
                    x, y = next_batch(args, samples, b)
                    #x = np.reshape(x, [])
                    time_next = extract_time(args, y)
                    if len(adj_list_prev) > 0:
                        adj_list_prev[0] = copy.copy(adj_list[-1])
                    adj_list = get_adjacency_list(x, adj_old, args.n)

                    if len(adj_list_prev) > 0:
                        adj_list_prev[1:] = copy.copy(adj_list[:-1])
                    else:
                        adj_list_prev = [np.zeros((args.n, args.n))]

                    #print "Debug adj list_prev:", adj_list_prev
                    #print "Debug adj_list:", adj_list

                    print "Debug adj_list_prev:"
                    for i_index in range(84):
                        print adj_list_prev[0][i_index]

                    print "Debug adj_list:"
                    for i_index in range(84):
                        print adj_list[0][i_index]

		    print "Debug adj_list ones:", len(adj_list), np.count_nonzero(adj_list[0])
		    print "Debug adj_list_prev ones:", len(adj_list_prev), np.count_nonzero(adj_list_prev[0])
		    adj_old = adj_list[-1]
                    print "Debug adj_old ones:", np.count_nonzero(adj_old)
		    features = get_one_hot_features(args.n)
		    print "Features", features
		    print dummy_features(adj_list[0], features, 5, 84, 84)
    '''
    model.train(args, data_train[:1])
