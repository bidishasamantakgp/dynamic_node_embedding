
import pickle
import tensorflow as tf
import numpy as np

def dummy_kl_gaussgauss(args, mu_1, sigma_1, mu_2, sigma_2):
        k = np.zeros([args.n])
        k.fill(args.z_dim)

        sigma_2_sigma_1 = []
        sigma_mu_sigma = []
        det = []
        for i in range(args.n):
                    sigma_2_inv = np.linalg.inv(sigma_2[i])
                    #print "Debug sigma_2_inv", sigma_2_inv
                    sigma_2_sigma_1.append(np.trace(np.multiply( sigma_2_inv, sigma_1[i])))
                    #print "Debug sigma_2_sigma_1:", sigma_2_sigma_1[i]
                    mu_diff = np.subtract(mu_2[i], mu_1[i])
                    #print "Debug mu_diff:", mu_diff
                    sigma_mu_sigma.append(np.matmul(np.matmul(np.transpose(mu_diff), sigma_2_inv), mu_diff))
                    #print "Debug sigma_mu_sigma:", sigma_mu_sigma[i]

                    det.append(np.log(np.maximum(np.linalg.det(sigma_2), 1e-09)) - np.log(np.maximum(np.linalg.det(sigma_1), 1e-09)))
        first_term = np.stack(sigma_2_sigma_1)
        second_term = np.stack(sigma_mu_sigma)
        third_term = np.stack(det)
        #print "Debug size", first_term.get_shape(), second_term.get_shape(), third_term.get_shape()
        #k = tf.fill([self.n], tf.cast(args.z_dim, tf.float32))
        return -np.sum(0.5 *(first_term + second_term + (third_term) - k))

def dummy_features(adj, feature, k, n, d):
    w_in = np.zeros([k, d, d])
    w_in.fill(0.5)
    #tf.get_variable(name="w_in", shape=[k,d,d], initializer=tf.constant_initializer(0.5))
    #w_in = tf.Print(w_in,[w_in], message="my w_in-values:")
    output_list = []

    for i in range(k):
            if i > 0:
                output_list.append( np.multiply(np.transpose(np.matmul(w_in[i], np.transpose(feature))), np.matmul(adj, output_list[i-1])))
            else:
                output_list.append(np.transpose(np.matmul(w_in[i], np.transpose(feature))))
            print("Debug output_list")
            print(output_list[i])

    return np.reshape(output_list[-1],[n, 1, d])

def extract_time(args, samples):
    time = []
    #for sample in samples:
    for i in range(args.seq_length):
        time.append(samples[0][i][2])
    time = np.resize(np.stack(time), (args.batch_size, args.seq_length, 1))
    return time


def get_one_hot_features(n):
    return np.ones([n, 1])
    #return np.identity(n)

def get_shape(tensor):
    '''return the shape of tensor as list'''
    return tensor.get_shape().as_list()

def load_data(filename):
    return pickle.load(open(filename, 'rb'))

def starting_adj(args, samples):
    input_data = load_data(args.data_file)
    adj = np.zeros([args.n, args.n])

    for i in range(4266):
    #for sample in samples:
        sample = input_data[i]
        u = int(sample[0] - 1)
        v = int(sample[1] - 1)
        m = sample[3]
        
        if m == 0:
            adj[u][v] = 1
            adj[v][u] = 1

    return adj

def create_samples(args):
    input_data = load_data(args.data_file)
    start = input_data[0][2]
    #start = 0
    input_data = [(int(u-1), int(v-1), ((t-start) * 1.0)/ (24 * 3600), m) for (u, v, t, m) in input_data]
    train_data = input_data[4266:args.train_size+4266]
    test_data = input_data[args.train_size+4266:]
    samples_train = []
    samples_test = []
    print(len(train_data))
    for i in range(0, len(train_data), args.seq_length):
        #if i + args.seq_length < len(train_data):

        sample = train_data[i : i + args.seq_length + 1]
        
        samples_train.append(sample)
        i += args.seq_length

    for i in range(0, len(test_data), args.seq_length):
        sample = test_data[i : i + args.seq_length + 1]
        samples_test.append(sample)
        i += args.seq_length

    return samples_train, samples_test

def get_adjacency_list(samples, adj, n):
    adj_list = []
    s = samples.shape[1]
    #s = len(samples)
    for i in range(s):
    #for sample in samples:
        
        sample = samples[0][i]
        u = int(sample[0])
        v = int(sample[1])
        m = sample[3]

        print("debug u, v", u, v)
        if m == 0:
            adj[u][v] = 1
            adj[v][u] = 1
        adj_list.append(adj)
    return adj_list

def next_batch(args, samples, i):
    reshaped = np.reshape(samples[i], [args.batch_size, args.seq_length+1, -1])
    #print reshaped.shape, samples[i]
    #x = samples[i][:-1]
    x = reshaped[:, :-1, :]
    #print "x", x.shape
    #y = samples[i][1:]
    y = reshaped[:, 1:, :]
    #print "y", y.shape
    return x, y


def print_vars(string):
    '''print variables in collection named string'''
    print("Collection name %s"%string)
    print("    "+"\n    ".join(["{} : {}".format(v.name, get_shape(v)) for v in tf.get_collection(string)]))
