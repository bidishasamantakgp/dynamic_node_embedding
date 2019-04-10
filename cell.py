from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class VAEGDCell(tf.nn.rnn_cell.RNNCell):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, features, bias_laplace, sample, eps, k, h_dim, x_dim, z_dim):
        '''
        Args:
        '''
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim
        self.sample = sample
        #self.name = self.__class__.__name__.lower()
        self.lstm_s = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        self.lstm_t = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)

    @property
    def state_size(self):
        return (self.n_h, self.n_h)

    @property
    def output_size(self):
        return self.n_h

    def __call__(self, x, state_temporal, state_structural, scope=None):
        
        '''
		Args:
			c_x - tensor to be filled up with random walk property
			n - number of nodes
			d - number of features in the feature matrix
			k - length of random walk
				defaults to be None
    	'''
        o_t, h_t = state_temporal
        o_s, h_s = state_structural
        
        n = self.adj[0].shape[0]
        d = self.features[0].shape[0]
        k = self.k

        with tf.variable_scope(scope or type(self).__name__):
            c_x = input_layer(c_x, self.adj, self.features, k, n, d, activation=None, batch_norm=False, istrain=False,
                              scope=None)

        with tf.variable_scope("Prior"):
            prior_hidden = fc_layer(tf.transpose(h_), self.n_prior_hidden, activation=tf.nn.relu, scope="hidden")
            prior_mu = fc_layer(prior_hidden, self.n_z, scope="mu")
            prior_mu = tf.reshape(prior_mu, [self.n, self.n_z, 1])
            prior_intermediate_sigma = fc_layer(prior_hidden, self.n_z, activation=tf.nn.softplus, scope="sigma")  # >=0
            prior_sigma = tf.matrix_diag(prior_intermediate_sigma, name="sigma")

        #c_x_1 = fc_layer(tf.matmul(self.bias_laplace, c_x[-1]), self.n_x_1, activation = tf.nn.relu, scope = "phi_c")# >=0
        #print("DEBUG shape prior", prior_hidden.shape, prior_intermediate_sigma.shape) 
        #print("Debug", c_x.shape, c_x_1.shape, tf.transpose(h).shape)
        
        with tf.variable_scope("Encoder"):
            (u, v) = current_event 
            a_uv = tf.gather_nd(association, (u,v))
            c_u = tf.gather(cluster, u)
            c_v = tf.gather(cluster, v)
            f_event = TBD

            input_concatenated = tf.concat(a_uv, c_u, c_v, f_event)
            enc_hidden = combined_layer(h_t, input_concatenated, activation=tf.nn.relu, scope="hidden")
            enc_mu = fc_layer(enc_hidden, self.n_z, activation=tf.nn.softplus, scope='mu')
            # output will be n X 1 then convert that to a diagonal matrix
            enc_intermediate_sigma = fc_layer(enc_hidden, self.n_z, activation=tf.nn.softplus, scope='sigma')
            enc_intermediate_sigma = tf.Print(enc_intermediate_sigma, [enc_intermediate_sigma], message="my debug_sigma-values:")
            enc_sigma = tf.matrix_diag(enc_intermediate_sigma, name="sigma")

        # Random sampling ~ N(0, 1)
        eps = self.eps
        
        # tf.random_normal((n, 5, 1), 0.0, 1.0, dtype=tf.float32)
        temp_stack = []
        for i in range(n):
            temp_stack.append(tf.matmul(enc_sigma[i], eps[i]))
        z = tf.add(enc_mu, tf.stack(temp_stack))

        # While we are trying to sample some edges, we sample Z from prior
        if self.sample:
            z = eps

        with tf.variable_scope("phi_z"):
            z_1 = fc_layer(tf.matmul(self.bias_laplace, tf.reshape(z, [n, self.n_z])), self.n_z_1, activation=tf.nn.relu)

        print("Debug2", c_x_1.shape, z_1.shape)

        with tf.variable_scope("Decoder"):
                z_stack = []
                for u in range(n):
                    # Assuming the graph is undirected
                    for v in range(n):

                        z_stack.append(tf.concat(values=(tf.transpose(z[u]), tf.transpose(z[v])), axis=1)[0])
                dec_hidden = fc_layer(tf.stack(z_stack), 1, activation=tf.nn.softplus, scope="hidden")

        output, state2 = self.lstm(tf.reshape(tf.concat(axis=1, values=(c_x_1, z_1)),[1, -1]), state)
        return (c_x, enc_mu, enc_sigma, enc_intermediate_sigma, dec_hidden, prior_mu, prior_sigma, prior_intermediate_sigma, z), state2

    def call(self, state, c_x, n, d, k, eps_passed, sample, bias_laplace):
        return self.__call__(state, c_x, n, d, k, eps_passed, sample, bias_laplace)
