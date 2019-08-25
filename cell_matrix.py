from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class MPPCell(object):

    """Variational Auto Encoder cell."""

    def __init__(self, args, features, B_old, r_old, eps):

        '''
        Args:
        '''
        self.eps = eps
        self.z_dim = args.z_dim
        self.n_c = args.n_c
        self.n = args.n
        self.n_h = args.h_dim

	#self.adj = adj
        #self.T = t_next
        #self.adj_prev = adj_prev

        self.d = args.d_dim
        self.k = args.k
        self.B_old = B_old
        self.r_old = r_old
        self.features = features
        self.sample = args.sample
        self.alpha = args.alpha
        self.lstm_s = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        self.lstm_t = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)


    def new_call(self, x, adj, adj_prev, t_next, state, scope=None):

        '''
        Args:
        x : input event (u,v,t, type)
        '''

        state_temporal, state_structural = state
        #print "Debug inside cell", x, state_temporal, state_structural
        #u, v, t, type_m = x
        u = x[0][0]
        v = x[0][1]
        t = x[0][2]
        type_m = x[0][3]



        h_t = state_temporal
        h_s = state_structural


        #o_t, h_t = state_temporal
        #o_s, h_s = state_structural

        #event_indicator = np.zeros([self.n])
        #event_indicator[u] = 1
        #event_indicator[v] = 1

        k = self.k

        with tf.variable_scope(scope or type(self).__name__, reuse=tf.AUTO_REUSE):
            scaling = tf.ones([self.n, 1])
            temp_u = tf.reshape(tf.one_hot(indices=u, depth=self.n), [1, self.n])
            temp_v = tf.reshape(tf.one_hot(indices=v, depth=self.n), [1, self.n])
            h_inter = fc_layer(h_t, self.n_h, scope="intermidiate")
            h_s_scaled = tf.matmul(scaling, h_s)
            h_t_scaled = tf.matmul(scaling, h_inter)

            val = np.array([(1 - type_m) * t])
            val = np.reshape(val, [1, -1])
            # print val.shape, val
            val_tf = tf.convert_to_tensor([[( 1 - type_m) * t]])
            t_scaled = tf.matmul(scaling, val)

            with tf.device('/gpu:0'):
             with tf.variable_scope("Prior"):
                
                y = input_layer(adj_prev, self.features, k, self.n, self.d)
                
                
                y_s = tf.concat([y, h_s_scaled], axis = 1)

                y_u = tf.matmul(temp_u, y)
                y_v = tf.matmul(temp_v, y)    

                u_append = tf.matmul(tf.matmul(adj_prev , tf.transpose(temp_u)), y_u)
                v_append = tf.matmul(tf.matmul(adj_prev , tf.transpose(temp_v)), y_v)

                append_val = tf.add(u_append, v_append)
               
                y_t = tf.concat([y, append_val, h_t_scaled])

                # Dimension is n X d
                #print "Debug Y", y_s_stack.shape, y_t_stack.shape
                prior_zeta_hidden = fc_layer(y_s, self.z_dim, activation=tf.nn.softplus, scope="zeta_hidden")
                prior_z_hidden = fc_layer(y_t, self.z_dim, activation=tf.nn.softplus, scope="z_hidden")

                prior_zeta_mu = fc_layer(prior_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_mu")
                prior_zeta_mu = tf.reshape(prior_zeta_mu, [self.n, self.z_dim, 1])

                prior_z_mu = fc_layer(prior_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_mu")
                prior_z_mu = tf.reshape(prior_z_mu, [self.n, self.z_dim, 1])

                prior_zeta_sigma = fc_layer(prior_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_sigma")  # >=0
                prior_zeta_sigma = tf.matrix_diag(prior_zeta_sigma)

                prior_z_sigma = fc_layer(prior_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_sigma")  # >=0
                prior_z_sigma = tf.matrix_diag(prior_z_sigma)

            with tf.device('/gpu:1'):
             with tf.variable_scope("Encoder"):
                
                y_current = input_layer(adj, self.features, k, self.n, self.d)

                y_s_enc = tf.concat([y, h_s_scaled, val_tf, t_scaled],  axis = 1)

                y_u_current = tf.matmul(temp_u, y_current)
                y_v_current = tf.matmul(temp_v, y_current)    

                u_append = tf.matmul(tf.matmul(adj , tf.transpose(temp_u)), y_u_current)
                v_append = tf.matmul(tf.matmul(adj , tf.transpose(temp_v)), y_v_current)

                append_val = tf.add(u_append, v_append)

                y_t_enc = tf.concat([y, append_val, h_t_scaled, t_scaled], axis = 1)

                enc_zeta_hidden = fc_layer( y_s_enc, self.z_dim,  scope="zeta_hidden" )
                enc_z_hidden = fc_layer( y_t_enc, self.z_dim,  scope="z_hidden" )

                enc_zeta_mu = fc_layer(enc_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_mu")
                enc_zeta_mu = tf.reshape(enc_zeta_mu, [self.n, self.z_dim, 1])

                enc_z_mu = fc_layer(enc_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_mu")
                enc_z_mu = tf.reshape(enc_z_mu, [self.n, self.z_dim, 1])

                enc_zeta_sigma = fc_layer( enc_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_sigma" )  # >=0
                enc_zeta_sigma = tf.matrix_diag(enc_zeta_sigma)

                enc_z_sigma = fc_layer( enc_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_sigma" )  # >=0
                enc_z_sigma = tf.matrix_diag( enc_z_sigma )

            with tf.device('/gpu:2'):
                # Random sampling ~ N(0, 1)
                eps = self.eps
                # At the time of training we use the posterior mu sigma
                z_stack = []
                zeta_stack = []
                for i in range(self.n):
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
            

            with tf.device('/gpu:3'):
             with tf.variable_scope("Decoder"):
                zeta_flatten = tf.reshape(zeta, [self.n, self.z_dim])
                c = fc_layer(zeta_flatten, self.n_c, activation=tf.nn.softplus, scope="cluster")
                B_hidden = fc_layer(tf.concat([h_s, tf.transpose(tf.gather(zeta, u)), tf.transpose(tf.gather(zeta, v))], axis = 1), self.n_c, activation=tf.nn.softplus, scope="SBM_hidden" )
                B = tf.add(tf.cast(type_m, tf.float32) * self.B_old, tf.matmul(scaling_b ,  tf.cast((1 - type_m), tf.float32) * B_hidden))
                r_hidden = fc_layer(tf.concat([h_t, tf.transpose(tf.gather(zeta, u)), tf.transpose(tf.gather(zeta, v))], axis=1), 1, activation=tf.nn.softplus, scope="R_hidden" )
                #print "Debug intermediate step", a.get_shape()                
                r = tf.add(tf.matmul(scaling, tf.cast(type_m, tf.float32) * r_hidden), tf.cast((1 - type_m), tf.float32) * self.r_old)
                P = tf.multiply(tf.subtract(tf.ones([self.n, self.n]), adj_prev), (self.alpha * tf.matmul(c, tf.matmul(B, tf.transpose(c))) + (1 - self.alpha) * r))
                time = tf.reshape(tf.cast([t_next - t], tf.float32),[1,1])
                time_scaled = tf.matmul(scaled, time)
                #       84 * 84 * 5
                l_a = fc_layer(tf.concat([tf.multiply(P, h_s_scaled), tf.multiply(P, time_scaled)], axis = 2), 1, activation=tf.nn.softplus, scope="association")
                z_trans = tf.transpose(z, [2, 1, 0])
                z_concat = tf.reshape(tf.matmul(z, z_trans), [0, 2, 1])
                l_c = fc_layer(tf.concat(z_concat, tf.matmul(scaled, tf.concat([h_t_scaled, time_scaled], axis = 2 ))), 1, activation=tf.nn.softplus, scope="communication")

                

            with tf.device('/gpu:4'):
             (o_t_new, h_t_new) = ([], h_t)
             time = tf.reshape(tf.cast([t], tf.float32),[1,1])
             #print "Debug z dim", tf.reduce_sum(z, axis=0).get_shape(), z.get_shape()
             z_reshape = tf.reshape(z,[self.n, 1, self.z_dim] )
             zeta_reshape = tf.reshape(zeta,[self.n, 1, self.z_dim])

             temp = tf.concat([tf.reduce_sum(z_reshape, axis=0), time], axis=1)
             #print "Temp", temp.get_shape(), h_t.get_shape(), tf.zeros([self.n_h]).get_shape(), h_s.get_shape()

             o_t_new, s_t = self.lstm_t(tf.concat([tf.reduce_sum(z_reshape, axis=0), time], axis=1), (tf.zeros([self.n_h]), h_t))
             o_s_new, s_s = self.lstm_s(tf.concat([tf.reduce_sum(zeta_reshape, axis=0), time], axis=1), (tf.zeros([self.n_h]), h_s))
             c, h_t_new = s_t
             c, h_s_new = s_s

        return (h_inter_enc, y_current, y_s_stack, y_s_enc_stack, y_t_enc_stack, enc_zeta_hidden, l_c, l_a, enc_zeta_mu, enc_z_mu, enc_zeta_sigma, enc_z_sigma, prior_zeta_mu, prior_z_mu, prior_zeta_sigma, prior_z_sigma, B, r), h_t_new, h_s_new

    def call(self, x, state):
        return self.__call__(x, state)
