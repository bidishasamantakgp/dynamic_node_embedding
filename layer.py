from utils import *
import tensorflow as tf


def input_layer(adj, feature, k, n, d, activation = None, batch_norm = False, istrain = False, scope = None):
    #with tf.device('/device:GPU:0'):
    with tf.variable_scope(scope or "input", reuse=tf.AUTO_REUSE):
        w_in = tf.get_variable(name="w_in", shape=[k,d,d], initializer=tf.constant_initializer(0.005))
        #w_in = tf.get_variable(name="w_in", shape = [k, d, d], initializer=tf.contrib.layers.xavier_initializer())
	#w_in = tf.Print(w_in,[w_in], message="my w_in-values:")
        output_list = []

        for i in range(k):
            if i > 0:
                #print "Debug I", i, output_list[0]
                #tf.matmul(w_in[i], tf.transpose(feature))
                #tf.matmul(adj, output_list[i-1])
                output_list.append( tf.multiply(tf.transpose(tf.matmul(w_in[i], tf.transpose(feature))),tf.matmul(adj, output_list[i-1])))
            else:
                output_list.append(tf.transpose(tf.matmul(w_in[i], tf.transpose(feature))))

        return tf.reshape(output_list[-1],[n, 1, d])
        #return tf.stack(output_list)


def fc_layer(input_, output_size, activation = None, batch_norm = False, istrain = False, scope = None, dropout=False):
    '''
    fully convlolution layer
    Args :
        input_  - 2D tensor
        output_size - list of the sizes for the output
            shape of output 2D tensor
        activation - activation function
            defaults to be None
        batch_norm - bool
            defaults to be False
            if batch_norm to apply batch_normalization
        istrain - bool
            defaults to be False
            indicator for phase train or not
        scope - string
            defaults to be None then scope becomes "fc"
    '''
    #with tf.device('/device:GPU:0'):
    with tf.variable_scope(scope or "fc", reuse=tf.AUTO_REUSE):
        #w = tf.get_variable(name="w", shape = [get_shape(input_)[1], output_size], initializer=tf.contrib.layers.xavier_initializer())
        w = tf.get_variable(name="w", shape = [get_shape(input_)[1], output_size], initializer=tf.constant_initializer(0.01))
        #w = tf.Print(w,[w], message="my W-values:")

        b = tf.get_variable(name="b", shape = [output_size], initializer=tf.constant_initializer(0.01))
        #b = tf.Print(b, [b], message="my B-values:"+scope)
        if activation is None:
            res = tf.nn.xw_plus_b(input_, w, b)
        else:
            res = activation(tf.nn.xw_plus_b(input_, w, b))

        if dropout:
            res = tf.nn.dropout(res, rate = 0.2)
        return res


def fc_layer_3d(input_, output_size, activation = None, batch_norm = False, istrain = False, scope = None, dropout=False):
    '''
    fully convlolution layer
    Args :
        input_  - 2D tensor
        output_size - list of the sizes for the output
            shape of output 2D tensor
        activation - activation function
            defaults to be None
        batch_norm - bool
            defaults to be False
            if batch_norm to apply batch_normalization
        istrain - bool
            defaults to be False
            indicator for phase train or not
        scope - string
            defaults to be None then scope becomes "fc"
    '''
    #with tf.device('/device:GPU:0'):
    with tf.variable_scope(scope or "fc", reuse=tf.AUTO_REUSE):
        #w = tf.get_variable(name="w", shape = [get_shape(input_)[1], output_size], initializer=tf.contrib.layers.xavier_initializer())
        w = tf.get_variable(name="w", shape = [get_shape(input_)[0], get_shape(input_)[2], output_size], initializer=tf.constant_initializer(0.01))
        #w = tf.get_variable(name="w", shape = [get_shape(input_)[0], get_shape(input_)[2], output_size], initializer=tf.contrib.layers.xavier_initializer())
        #tf.tile(w, )
        #w = tf.Print(w,[w], message="my W-values:")

        b = tf.get_variable(name="b", shape = [output_size], initializer=tf.constant_initializer(0.01))
        #b = tf.Print(b, [b], message="my B-values:"+scope)
        if activation is None:
            res = tf.nn.xw_plus_b(input_, w, b)
        else:
            res = activation(tf.nn.xw_plus_b(input_, w, b))

        if dropout:
            res = tf.nn.dropout(res, rate = 0.2)
        return res
