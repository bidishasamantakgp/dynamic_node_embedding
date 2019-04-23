from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class MPPCell(tf.nn.rnn_cell.RNNCell):
    """Variational Auto Encoder cell."""

    def __init__(self, adj_prev, features, bias_laplace, sample, eps, k, h_dim, x_dim, z_dim):
        
        '''
        Args:
        '''
        
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.n_c = no_cluster
        sellf.n = n
        self.adj = adj_prev
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
        event_indicator = np.zeros([self.n])
        event_indicator[u] = 1
        event_indicator[v] = 1
        k = self.k

        with tf.variable_scope(scope or type(self).__name__):
        
            
            with tf.variable_scope("Prior"):
                h_inter = fc_layer(h_t, self.dim_z, scope="intermidiate")
                y_s = []
                y_t = []
                for i range(self.n):
                    y_s.append(tf.concat(y[i], hs))
                    y_t.append(tf.add(tf.concat(y[i], tf.matmul(tf.multiply(self.adj[i], event_indicator), y), h_inter), tf.concat(tf.zeros([1, 2* self.f_size]),h_inter)))

                # Dimension is n X d
                prior_zeta_hidden = fc_layer(y_s, self.dim_z, activation=tf.nn.relu, scope="zeta_hidden")
                prior_z_hidden = fc_layer(y_t, self.dim_z, activation=tf.nn.relu, scope="z_hidden")
            
                prior_zeta_mu = fc_layer(prior_zeta_hidden, self.n_z, activation=tf.nn.softplus scope="zeta_mu")
                prior_zeta_mu = tf.reshape(prior_zeta_mu, [self.n, self.n_z, 1])

                prior_z_mu = fc_layer(prior_z_hidden, self.n_z, activation=tf.nn.softplus scope="z_mu")
                prior_z_mu = tf.reshape(prior_z_mu, [self.n, self.n_z, 1])

                prior_zeta_sigma = fc_layer(prior_zeta_hidden, self.n_z, activation=tf.nn.softplus, scope="zeta_sigma")  # >=0
                prior_zeta_sigma = tf.matrix_diag(prior_zeta_sigma)

                prior_z_sigma = fc_layer(prior_z_hidden, self.n_z, activation=tf.nn.softplus, scope="z_sigma")  # >=0
                prior_z_sigma = tf.matrix_diag(prior_z_sigma)
            

            with tf.variable_scope("Encoder"):
                # (i-1)th input size N x d
                self.adj[u][v] = 1
                self.adj[v][u] = 1
                y_current = input_layer(self.adj)
                y_s = []
                y_t = []
                h_inter = fc_layer(h_t, self.dim_z, scope="intermidiate")
                for i range(self.n):
                    y_s.append(tf.concat(y_current[i], h_s, (1-type_m) * t))
                    y_t.append(tf.concat(tf.add(tf.concat(y_current[i], tf.matmul(tf.multiply(self.adj[i], event_indicator), y), h_inter), tf.concat(tf.zeros([1, 2* self.f_size]),h_inter))), type_m * t)

                enc_zeta_hidden = fc_layer(y_s, self.dim_z, activation=tf.nn.relu, scope="zeta_hidden")
                enc_z_hidden = fc_layer(y_t, self.dim_z, activation=tf.nn.relu, scope="z_hidden")

                enc_zeta_mu = tf.reshape(enc_zeta_mu, [self.n, self.n_z, 1])
                enc_z_mu = tf.reshape(enc_z_mu, [self.n, self.n_z, 1])
                
                enc_zeta_sigma = fc_layer(enc_zeta_hidden, self.n_z, activation=tf.nn.softplus, scope="zeta_sigma")  # >=0
                enc_zeta_sigma = tf.matrix_diag(enc_zeta_sigma)

                enc_z_sigma = fc_layer(enc_z_hidden, self.n_z, activation=tf.nn.softplus, scope="z_sigma")  # >=0
                enc_z_sigma = tf.matrix_diag(enc_z_sigma)
    

            # Random sampling ~ N(0, 1)
            eps = self.eps
            # At the time of training we use the posterior mu sigma
            z_stack = []
            zeta_stack = []
            for i in range(n):
                z_stack.append(tf.matmul(enc_z_sigma[i], eps[i]))
                zeta_stack.append(tf.matmul(enc_zeta_sigma[i], eps[i]))
            
            z = tf.add(enc_z_mu, tf.stack(z_stack))
            zeta = tf.add(enc_zeta_mu, tf.stack(zeta_stack))
            # After training when we want to generate values:
            if self.sample: 
                z_stack = []
                zeta_stack = []
                for i in range(n):
                    z_stack.append(tf.matmul(prior_z_sigma[i], eps[i]))
                    zeta_stack.append(tf.matmul(prior_zeta_sigma[i], eps[i]))
                z = tf.add(enc_z_mu, tf.stack(z_stack))
                zeta = tf.add(enc_zeta_mu, tf.stack(zeta_stack))
            
            with tf.variable_scope("Decoder"):
                c = fc_layer(zeta, self.n_c, activation=tf.nn.softplus, scope="cluster")
                B_hidden = fc_layer(tf.concat(h_s, zeta[u], zeta[v]), self.n_c,activation=tf.nn.softplus, scope="SBM_hidden" )
                B = type_m * B + (1 - type_m) * B_hidden
                r_hidden = fc_layer(tf.concat(h_t, zeta[u], zeta[v]), self.n_c,activation=tf.nn.softplus, scope="R_hidden" )
                r = (1 - type_m) * r + type_m * r_hidden 
                P = tf.multiply(tf.subtract(tf.ones(self.n, self.n), self.adj), (alpha * tf.matmul(tf.matmul(tf.transpose(c), B), c) + (1 - alpha) * r))
                for i in range(self.n):
                    for j in range(self.n):
                        p_uv = tf.gather_nd(P, (i,j))
                        lambda_association.append(tf.concat(p_uv * h_s, p_uv * (self.T - t)))
                        lambda_communication.append(tf.concat(h_t, (self.T - t), z[i], z[j]))

            (o_t_new, h_t_new) = (o_t, h_t)
            if type_m == 1:
                o_t_new, h_t_new = self.lstm_t(tf.concat(tf.sum(z), t),h_t)            
            
            (o_s_new, h_s_new) = (o_s, h_s)
            if type_m == 0:
                o_s_new, h_s_new= self.lstm_s(tf.concat(tf.sum(zeta), t), h_s)
             
        return (lambda_communication, lambda_association, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma), (h_t_new, h_s_new)

    def call(self, x, state_temporal, state_structural):
        return self.__call__(x, state_temporal, state_structural)
