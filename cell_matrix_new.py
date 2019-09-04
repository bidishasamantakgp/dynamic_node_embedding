from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class MPPCell(tf.contrib.rnn.RNNCell):

    """Variational Auto Encoder cell."""

    def __init__(self, args, features, eps):
        '''
        Args:
        '''
        self.eps = eps
        self.z_dim = args.z_dim
        self.n_c = args.n_c
        self.n = args.n
        self.n_h = args.h_dim

        self.d = args.d_dim
        self.k = args.k
        self.features = features
        self.sample = args.sample
        self.alpha = args.alpha
        self.lstm_s = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        self.lstm_t = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)


    def new_call(self, x, state, scope=None):

        '''
        Args:
        x : input event (u,v,t, type)
        '''
        h, c = state
        state_temporal, state_structural = tf.split(h, [self.z_dim, self.z_dim], 1)
        
        
        u = x[0][0]
        v = x[0][1]
        t = x[0][2] 
        type_m = x[0][3]
        
        B_old = x[1]
        r_old = x[2]
        t_next = x[3] 


        h_t = state_temporal
        h_s = state_structural
        k = self.k

        with tf.variable_scope(scope or type(self).__name__, reuse=tf.AUTO_REUSE):
            scaling = tf.ones([self.n, 1], tf.float32)
            scaling_b = tf.ones([self.n_c, 1], tf.float32)
            scaling_n2 = tf.ones([self.n, self.n, 1], tf.float32)
            temp_u = tf.reshape(tf.one_hot(indices=u, depth=self.n), [1, self.n])
            temp_v = tf.reshape(tf.one_hot(indices=v, depth=self.n), [1, self.n])
            h_inter = fc_layer(h_t, self.n_h, scope="intermidiate")
            h_s_scaled = tf.matmul(scaling, h_s)
            h_t_scaled = tf.matmul(scaling, h_inter)

            val_tf = tf.convert_to_tensor([[( 1 - type_m) * t]])
            t_scaled = tf.matmul(scaling, tf.cast(val_tf, tf.float32))

            #with tf.device('/gpu:0'):
            with tf.variable_scope("Prior"):
                
                y = tf.reshape(input_layer(adj_prev, self.features, k, self.n, self.d), [self.n, -1])
                y_s = tf.concat([y, h_s_scaled], axis = 1)
                y_t = tf.concat([y, h_t_scaled], axis=1)
                
                # Dimension is n X d
                prior_zeta_hidden = fc_layer(y_s, self.z_dim, activation=tf.nn.softplus, scope="zeta_hidden")
                prior_z_hidden = fc_layer(y_t, self.z_dim, activation=tf.nn.softplus, scope="z_hidden")

                prior_zeta_mu = fc_layer(prior_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_mu")
                prior_zeta_mu = tf.reshape(prior_zeta_mu, [self.n, self.z_dim, 1])

                prior_z_mu = fc_layer(prior_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_mu")
                prior_z_mu = tf.reshape(prior_z_mu, [self.n, self.z_dim, 1])

                prior_zeta_sigma_dia = fc_layer(prior_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_sigma")  # >=0
                prior_zeta_sigma = tf.matrix_diag(prior_zeta_sigma_dia)

                prior_z_sigma_dia = fc_layer(prior_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_sigma")  # >=0
                prior_z_sigma = tf.matrix_diag(prior_z_sigma_dia)

            #with tf.device('/gpu:1'):
            with tf.variable_scope("Encoder"):
                
                y_current = tf.reshape(input_layer(adj, self.features, k, self.n, self.d), [self.n, -1])
                y_s_enc = tf.concat([y_current, h_s_scaled, t_scaled],  axis = 1)

                y_u_current = tf.matmul(temp_u, y_current)
                y_v_current = tf.matmul(temp_v, y_current)    

                u_append = tf.matmul(tf.matmul(adj , tf.transpose(temp_u)), y_u_current)
                v_append = tf.matmul(tf.matmul(adj , tf.transpose(temp_v)), y_v_current)

                append_val = tf.add(u_append, v_append)

                y_t_enc = tf.concat([y_current, append_val, h_t_scaled, t_scaled], axis = 1)

                enc_zeta_hidden = fc_layer( y_s_enc, self.z_dim,  scope="zeta_hidden" )
                enc_z_hidden = fc_layer( y_t_enc, self.z_dim,  scope="z_hidden" )

                enc_zeta_mu = fc_layer(enc_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_mu")
                enc_zeta_mu = tf.reshape(enc_zeta_mu, [self.n, self.z_dim, 1])

                enc_z_mu = fc_layer(enc_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_mu")
                enc_z_mu = tf.reshape(enc_z_mu, [self.n, self.z_dim, 1])

                enc_zeta_sigma_dia = fc_layer( enc_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_sigma" )  # >=0
                enc_zeta_sigma = tf.matrix_diag(enc_zeta_sigma_dia)

                enc_z_sigma_dia = fc_layer( enc_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_sigma" )  # >=0
                enc_z_sigma = tf.matrix_diag( enc_z_sigma_dia )

            #with tf.device('/gpu:2'):
            # Random sampling ~ N(0, 1)
            eps = self.eps
            # At the time of training we use the posterior mu sigma
            # eps_reshape = tf.transpose(eps, [0, 2, 1])
            z = tf.add(enc_z_mu, tf.matmul(enc_z_sigma, eps))
            zeta = tf.add(enc_zeta_mu, tf.matmul(enc_zeta_sigma, eps))
            
            # After training when we want to generate values:
            if self.sample:
                    z = tf.add(prior_z_mu, tf.matmul(prior_z_sigma, eps))
                    zeta = tf.add(prior_zeta_mu, tf.matmul(prior_zeta_sigma, eps))

            #with tf.device('/gpu:3'):
            with tf.variable_scope("Decoder"):
                zeta_flatten = tf.reshape(zeta, [self.n, self.z_dim])
                C = tf.nn.softmax(fc_layer(zeta_flatten, self.n_c, scope="cluster"), axis=1)
                B_hidden = fc_layer(tf.concat([h_s, tf.transpose(tf.gather(zeta, u)), tf.transpose(tf.gather(zeta, v))], axis = 1), self.n_c, activation=tf.nn.softplus, scope="SBM_hidden" )
                B = tf.add(tf.cast(type_m, tf.float32) * B_old, tf.matmul(scaling_b ,  tf.cast((1 - type_m), tf.float32) * B_hidden))
                r_hidden = fc_layer(tf.concat([h_t, tf.transpose(tf.gather(zeta, u)), tf.transpose(tf.gather(zeta, v))], axis=1), 1, activation=tf.nn.softplus, scope="R_hidden" )
                #print "Debug intermediate step", a.get_shape()                
                
                r = tf.add(tf.matmul(scaling, tf.cast(type_m, tf.float32) * r_hidden), tf.cast((1 - type_m), tf.float32) * r_old)
                P = tf.multiply(tf.subtract(tf.ones([self.n, self.n]), adj_prev), (self.alpha * tf.matmul(C, tf.matmul(B, tf.transpose(C))) + (1 - self.alpha) * r))
                time = tf.reshape(tf.cast([t_next - t], tf.float32),[1,1])
                
                #   84 * 84 * 5
                P = tf.reshape(P, [self.n, self.n, 1])
                
                list_p = []
                for i in range(self.n_h):
                    list_p.append(P)

                P_concat = tf.concat(list_p, axis=2)
                h_s_scaled = tf.matmul(scaling, h_s)
                h_s_scaled_n2 = tf.reshape(tf.tile(h_s_scaled,[self.n, 1]), [self.n, self.n, self.n_h])
                h_t_scaled_n2 = tf.reshape(tf.tile(h_t_scaled,[self.n, 1]), [self.n, self.n, self.n_h])
                time_scaled = tf.matmul(scaling, time)
                time_scaled_n2 = tf.reshape(tf.tile(t_scaled,[self.n, 1]), [self.n, self.n, 1])


                l_a = fc_layer_3d(tf.concat([tf.multiply(P_concat, h_s_scaled_n2), tf.multiply(P, time_scaled_n2)], axis = 2), 1, activation=tf.nn.softplus, scope="association")
                z_reshape = tf.reshape(z, [self.n, -1])
                z_concat = tf.reshape(tf.matmul(z_reshape, tf.transpose(z_reshape)), [self.n, self.n, 1])

                l_c = fc_layer_3d(tf.concat([h_t_scaled_n2, time_scaled_n2, z_concat], axis = 2 ), 1, activation=tf.nn.softplus, scope="communication")
                

            #with tf.device('/gpu:4'):
            (o_t_new, h_t_new) = ([], h_t)
            time = tf.reshape(tf.cast([t], tf.float32),[1,1])
            #print "Debug z dim", tf.reduce_sum(z, axis=0).get_shape(), z.get_shape()
            z_reshape = tf.reshape(z,[self.n, 1, self.z_dim] )
            zeta_reshape = tf.reshape(zeta,[self.n, 1, self.z_dim])

            temp = tf.concat([tf.reduce_sum(z_reshape, axis=0), time], axis=1)
            #print "Temp", temp.get_shape(), h_t.get_shape(), tf.zeros([self.n_h]).get_shape(), h_s.get_shape()

            o_t_new, s_t = self.lstm_t(tf.concat([tf.reduce_sum(z_reshape, axis=0), time], axis=1), (c, h_t))
            o_s_new, s_s = self.lstm_s(tf.concat([tf.reduce_sum(zeta_reshape, axis=0), time], axis=1), (c, h_s))
            c, h_t_new = s_t
            c, h_s_new = s_s

            state_concat = tf.concat([h_t_new, h_s_new], axis=1)

        return (h_inter, y_current, y_s, y_s_enc, y_t_enc, enc_zeta_hidden, l_c, l_a, enc_zeta_mu, enc_z_mu, enc_zeta_sigma_dia, enc_z_sigma_dia, prior_zeta_mu, prior_z_mu, prior_zeta_sigma_dia, prior_z_sigma_dia, B, r, P, C), (c, state_concat)

    def call(self, x, state):
        return self.__call__(x, state)
