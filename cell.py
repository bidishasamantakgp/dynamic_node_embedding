from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class MPPCell(tf.nn.rnn_cell.RNNCell):
    """Variational Auto Encoder cell."""

    def __init__(self, adj, features, bias_laplace, sample, eps, k, h_dim, x_dim, z_dim):
        '''
        Args:
        '''
        self.n_enc_hidden = z_dim
        self.n_dec_hidden = x_dim
        self.n_prior_hidden = z_dim

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
			x : input event (u,v,t, type)
    	'''
        (u, v, t, type_m) = x
        o_t, h_t = state_temporal
        o_s, h_s = state_structural
        
        k = self.k

        with tf.variable_scope(scope or type(self).__name__):
        
            # if association event the graph structure changes
        
            
            zs = zs_prev
            with tf.variable_scope("Prior"):
                #TODO : Needs to clarify
                prior_hidden_1 = fc_layer(tf.transpose(h_t), comp_adj, self.n_prior_hidden, activation=tf.nn.relu, scope="hidden1")   
                prior_hidden_2 = fc_layer(zs, self.n_prior_hidden, activation=tf.nn.relu, scope="hidden2")

                prior_mu = fc_layer(tf.concat(prior_hidden_1, prior_hidden_2), self.n_z, activation=tf.nn.softplus scope="mu")
                prior_mu = tf.reshape(prior_mu, [self.n, self.n_z, 1])
                prior_intermediate_sigma = fc_layer(prior_hidden, self.n_z, activation=tf.nn.softplus, scope="sigma")  # >=0
                prior_sigma = tf.matrix_diag(prior_intermediate_sigma, name="sigma")
            
            if type == "association":
                zs = input_embedding(zs_prev, h_s)
            

            with tf.variable_scope("Encoder"):
                cluster_hidden = fc_layer(zs, 1, activation=tf.nn.softplus, scope="cluster")
                cluster = tf.softmax(cluster_hidden)
                a_uv = tf.gather_nd(adj, (u,v))
                c_u = tf.gather(cluster, u)
                c_v = tf.gather(cluster, v)
                B = fc_layer(tf.concat(h_s, c_u, c_v), n_users, activation=tf.nn.softplus, scope="B")
                # Which information is required????
                f_event = TBD
                input_concatenated = tf.concat(a_uv, c_u, c_v, f_event)

                enc_hidden = fc_layer(h_t, input_concatenated, activation=tf.nn.relu, scope="hidden")
                enc_mu = fc_layer(enc_hidden, self.n_z, activation=tf.nn.softplus, scope='mu')
                # output will be n X 1 then convert that to a diagonal matrix
                enc_intermediate_sigma = fc_layer(enc_hidden, self.n_z, activation=tf.nn.softplus, scope='sigma')
                enc_intermediate_sigma = tf.Print(enc_intermediate_sigma, [enc_intermediate_sigma], message="my debug_sigma-values:")
                enc_sigma = tf.matrix_diag(enc_intermediate_sigma, name="sigma")


            # Random sampling ~ N(0, 1)
            eps = self.eps
            # At the time of training we use the posterior mu sigma
            z = tf.add(tf.matmul(enc_sigma, eps), enc_mu)
            # After training when we want to generate values:
            if sample: 
                z = tf.add(tf.matmul(prior_sigma, eps), prior_mu)

            # While we are trying to sample some edges, we sample Z from prior
            if self.sample:
                z = eps
        
            with tf.variable_scope("Decoder"):
                Alpha_hidden = fc_layer(tf.concat(h_t, tf.gather(z_s, u), tf.gather(z_s, v), scope="alpha_hidden")
                Alpha = p * tf.matmul(tf.matmul(tf.transpose(C), B), C) + (1 - p) * Alpha_hidden
                TAO = fc_layer(Alpha * (1- T + b), 1, activation="relu")
                delT = 
                u_next =
                v_next = 

                
            
            (o_t_new, h_t_new) = (o_t, h_t)
            if type == "communication":
                o_t_new, h_t_new = self.lstm_t(tf.concat(tf.sum(z), t),h_t)            
            
            (o_s_new, h_s_new) = (o_s, h_s)
            if type == "association":
                o_s_new, h_s_new= self.lsmt_s(t,h_s)
        #TBD         
        return (), 

    def call(self, state, c_x, n, d, k, eps_passed, sample, bias_laplace):
        return self.__call__()
