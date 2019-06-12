
import pickle
import tensorflow as tf
import numpy as np

def extract_time(samples):
    time = []
    for sample in samples:
        time.append(sample[2])
    return time


def get_one_hot_features(n):
    return np.identity(n)

def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def load_data(filename):
    return pickle.load(open(filename))

def create_samples(args):
    input_data = load_data(args.data_file)
    train_data = input_data[:args.train_size]
    test_data = input_data[args.train_size:]
    samples_train = []
    samples_test = []

    for i in range(0, len(train_data), args.seq_length):
        sample = train_data[i : i + args.seq_length + 1]
        samples_train.append(sample)
    
    for i in range(0, len(test_data), args.seq_length):
        sample = test_data[i : i + args.seq_length + 1]
        samples_test.append(sample)

    return samples_train, samples_test

def get_adj_list(samples, n):
    
    adj = np.zeros((n, n))
    adj_list = []

    for sample in samples:
        u = sample[0]
        v = sample[1]
        m = sample[3]

        if m == 0:
            adj[u][v] = 1
            adj[v][u] = 1
        adj_list.append(adj)
    return adj_list

def next_batch(args, samples, i):
    reshaped = np.reshape(samples[i], len(samples[i]), 1)
    x = reshaped[:,-1]
    y = reshaped[:1:]
    return x, y


def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print("    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in tf.get_collection(string)]))
