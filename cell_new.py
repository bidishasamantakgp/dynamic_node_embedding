from layer import *
import tensorflow as tf
import numpy as np
from utils import *
from math import exp


class MPPCell(object):
    """Variational Auto Encoder cell."""

    def __init__(self, args, adj, adj_prev, features, B_old, r_old, eps):
        
        '''
        Args:
        '''
        self.eps = eps
        self.z_dim = args.z_dim
        self.n_c = args.n_c
        self.n = args.n
        self.n_h = args.h_dim
        self.adj = adj
        #self.T = t_next
        self.adj_prev = adj_prev
        self.d = args.d_dim
        self.k = args.k
        self.B_old = B_old
        self.r_old = r_old
        self.features = features
        self.sample = args.sample
        self.alpha = args.alpha
        self.lstm_s = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)
        self.lstm_t = tf.contrib.rnn.LSTMCell(self.n_h, state_is_tuple=True)


    def new_call(self, x, t_next, state, scope=None):
        
        '''
	Args:
	    x : input event (u,v,t, type)
    	'''
        
        state_temporal, state_structural = state
        print "Debug inside cell", x, state_temporal, state_structural
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
            with tf.variable_scope("Prior"):
                h_inter = fc_layer(h_t, self.n_h, scope="intermidiate")
                y = input_layer(self.adj_prev, self.features, k, self.n, self.d)
                y_s = []
                y_t = []
                for i in range(self.n):
                    print "Debug y_i", y[i].dtype, h_s.dtype, y.get_shape(), h_s.get_shape()
                    y_s.append(tf.concat([y[i], h_s], axis=1))
                    print "Debug y_s shape", y_s[i].get_shape()
                    temp1 = tf.gather_nd(self.adj_prev, (i, u))
                    temp2 = tf.gather(y, u)
                    temp3 = tf.gather_nd(self.adj_prev, (i,v))
                    temp4 = tf.gather(y, v)
                    temp5 = tf.multiply(temp1, temp2)
                    temp6 = tf.multiply(temp3, temp4)
                    intermediate = tf.add_n([temp5, temp6])
                    print "Debug intermediate size", intermediate.get_shape(), h_inter.get_shape()
                    #intermediate = tf.add_n(tf.multiply(tf.gather_nd(self.adj, (i, u)), tf.gather(y, u)), tf.multiply(tf.gather_nd(self.adj, (i,v)), tf.gather(y, v)))
                    first = tf.concat([y[i], intermediate, h_inter], axis=1)
                    second = tf.concat([tf.zeros([1, 2 * self.d]),h_inter], axis=1)

                    y_t.append(tf.add(first, second))


                y_s_stack = tf.reshape(tf.stack(y_s), [self.n,-1])
                y_t_stack = tf.reshape(tf.stack(y_t), [self.n, -1])

                # Dimension is n X d
                print "Debug Y", y_s_stack.shape, y_t_stack.shape
                prior_zeta_hidden = fc_layer(y_s_stack, self.z_dim, activation=tf.nn.relu, scope="zeta_hidden")
                prior_z_hidden = fc_layer(y_t_stack, self.z_dim, activation=tf.nn.relu, scope="z_hidden")
            
                prior_zeta_mu = fc_layer(prior_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_mu")
                prior_zeta_mu = tf.reshape(prior_zeta_mu, [self.n, self.z_dim, 1])

                prior_z_mu = fc_layer(prior_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_mu")
                prior_z_mu = tf.reshape(prior_z_mu, [self.n, self.z_dim, 1])

                prior_zeta_sigma = fc_layer(prior_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_sigma")  # >=0
                prior_zeta_sigma = tf.matrix_diag(prior_zeta_sigma)

                prior_z_sigma = fc_layer(prior_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_sigma")  # >=0
                prior_z_sigma = tf.matrix_diag(prior_z_sigma)
            

            with tf.variable_scope("Encoder"):
                # (i-1)th input size N x d
                # self.adj[u][v] = 1
                # self.adj[v][u] = 1
                
                y_current = input_layer(self.adj, self.features, k, self.n, self.d)
                
                y_s = []
                y_t = []
                
                h_inter = fc_layer(h_t, self.n_h, scope="intermidiate")
                
                for i in range(self.n):
                    val = np.array([(1 - type_m) * t])
                    val = np.reshape(val, [1, -1])
                    print val.shape, val
                    val_tf = tf.convert_to_tensor([[( 1 - type_m) * t]])
                    print "Debu val_tf", val_tf.get_shape(), val_tf
                    y_s.append(tf.concat([y_current[i], h_s, tf.dtypes.cast(val_tf, tf.float32)], axis = 1))
                    temp1 = tf.gather_nd(self.adj, (i, u))
                    temp2 = tf.gather(y_current, u)
                    temp3 = tf.gather_nd(self.adj, (i,v))
                    temp4 = tf.gather(y_current, v)
                    temp5 = tf.multiply(temp1, temp2)
                    temp6 = tf.multiply(temp3, temp4)
                    intermediate = tf.add_n([temp5, temp6])
                    print "Debug intermediate size", intermediate.get_shape(), h_inter.get_shape()
                    #intermediate = tf.add_n(tf.multiply(tf.gather_nd(self.adj, (i, u)), tf.gather(y, u)), tf.multiply(tf.gather_nd(self.adj, (i,v)), tf.gather(y, v)))
                    first = tf.concat([y[i], intermediate, h_inter], axis=1)
                    second = tf.concat([tf.zeros([1, 2 * self.d]),h_inter], axis=1)
                    m_t = tf.convert_to_tensor([[type_m * t]])
                    y_t.append(tf.concat([tf.add(first, second), tf.dtypes.cast(m_t, tf.float32)], axis = 1))

                y_s_stack = tf.reshape(tf.stack(y_s), [self.n,-1])
                y_t_stack = tf.reshape(tf.stack(y_t), [self.n, -1])
                print "Debug Y", y_s_stack.shape, y_t_stack.shape

                enc_zeta_hidden = fc_layer( y_s_stack, self.z_dim, activation=tf.nn.relu, scope="zeta_hidden" )
                enc_z_hidden = fc_layer( y_t_stack, self.z_dim, activation=tf.nn.relu, scope="z_hidden" )

                enc_zeta_mu = fc_layer(enc_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_mu")
                enc_zeta_mu = tf.reshape(enc_zeta_mu, [self.n, self.z_dim, 1])

                enc_z_mu = fc_layer(enc_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_mu")
                enc_z_mu = tf.reshape(enc_z_mu, [self.n, self.z_dim, 1])
                
                enc_zeta_sigma = fc_layer( enc_zeta_hidden, self.z_dim, activation=tf.nn.softplus, scope="zeta_sigma" )  # >=0
                enc_zeta_sigma = tf.matrix_diag(enc_zeta_sigma)

                enc_z_sigma = fc_layer( enc_z_hidden, self.z_dim, activation=tf.nn.softplus, scope="z_sigma" )  # >=0
                enc_z_sigma = tf.matrix_diag( enc_z_sigma )
    

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
            
            with tf.variable_scope("Decoder"):
                zeta_flatten = tf.reshape(zeta, [self.n, self.z_dim]) 
                c = fc_layer(zeta_flatten, self.n_c, activation=tf.nn.softplus, scope="cluster")
                
                B_hidden = fc_layer(tf.concat([h_s, tf.transpose(tf.gather(zeta, u)), tf.transpose(tf.gather(zeta, v))], axis = 1), self.n_c, activation=tf.nn.softplus, scope="SBM_hidden" )
                B = tf.cast(type_m, tf.float32) * self.B_old + tf.cast((1 - type_m), tf.float32) * B_hidden
                r_hidden = fc_layer(tf.concat([h_t, tf.transpose(tf.gather(zeta, u)), tf.transpose(tf.gather(zeta, v))], axis=1), 1, activation=tf.nn.softplus, scope="R_hidden" )

                a = tf.cast(type_m, tf.float32) * r_hidden
                print "Debug intermediate step", a.get_shape()
                second = tf.fill(value = a[0][0], dims = [self.n, self.n])

                first = tf.multiply(tf.fill(value = tf.cast(1 - type_m, tf.float32) , dims = [self.n, self.n]), self.r_old)
                print "Debug first", first.get_shape(), second.get_shape()
                r = tf.add(first, second)
                print "Debug r size", r.get_shape() 
                P = tf.multiply(tf.subtract(tf.ones([self.n, self.n]), self.adj), (self.alpha * tf.matmul(c, tf.matmul(B, tf.transpose(c))) + (1 - self.alpha) * r))
                lambda_association = []
                lambda_communication = []
                for i in range(self.n):
                    for j in range(self.n):
                        #p_uv = (1.0 - tf.gather_nd(adj, (i,j))) * (self.alpha * tf.gather())
                        p_uv = P[i][j]
                        #p_uv = tf.gather_nd(P, [i,j])
                        time = tf.reshape(tf.cast([t_next - t], tf.float32),[1,1])
                        #print "Debug time", time.get_shape()
                        lambda_association.append(tf.concat([p_uv * h_s, p_uv * time], axis=1))
                        lambda_communication.append(tf.concat([h_t, time, tf.transpose(z[i]), tf.transpose(z[j])], axis = 1))
                l_a = fc_layer(tf.reshape(tf.stack(lambda_association), [self.n * self.n, -1]), 1, activation=tf.nn.softplus, scope="association")
                l_c = fc_layer(tf.reshape(tf.stack(lambda_communication), [self.n * self.n, -1]), 1, activation=tf.nn.softplus, scope="communication")

            (o_t_new, h_t_new) = ([], h_t)
            time = tf.reshape(tf.cast([t], tf.float32),[1,1])
            print "Debug z dim", tf.reduce_sum(z, axis=0).get_shape(), z.get_shape()
            z_reshape = tf.reshape(z,[self.n, 1, self.z_dim] )
            zeta_reshape = tf.reshape(zeta,[self.n, 1, self.z_dim])
            
            temp = tf.concat([tf.reduce_sum(z_reshape, axis=0), time], axis=1)
            print "Temp", temp.get_shape(), h_t.get_shape(), tf.zeros([self.n_h]).get_shape(), h_s.get_shape()
            
            o_t_new, s_t = self.lstm_t(tf.concat([tf.reduce_sum(z_reshape, axis=0), time], axis=1), (tf.zeros([]), h_t))
            o_s_new, s_s = self.lstm_s(tf.concat([tf.reduce_sum(zeta_reshape, axis=0), time], axis=1), (tf.zeros([self.n_h]), h_s)) 
            c, h_t_new = s_t
            c, h_s_new = s_s
             
        return (l_c, l_a, enc_zeta_mu, enc_z_mu, enc_zeta_sigma, enc_z_sigma, prior_zeta_mu, prior_z_mu, prior_zeta_sigma, prior_z_sigma), h_t_new, h_s_new

    def call(self, x, state):
        return self.__call__(x, state)
