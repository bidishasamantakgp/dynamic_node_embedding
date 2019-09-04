import tensorflow as tf                
import numpy as np

k = tf.fill([84], tf.cast(15, tf.float32))

#with tf.variable_scope("kl_gaussisan"):

                #diag_mat_1 = tf.diag_part(sigma_1)
                #diag_mat_2 = tf.diag_part(sigma_2)

sigma_1 = tf.placeholder(dtype=tf.float32, shape=[84, 15])
sigma_2 = tf.placeholder(dtype=tf.float32, shape=[84, 15])

mu_1 = tf.placeholder(dtype=tf.float32, shape=[84, 15, 1])
mu_2 = tf.placeholder(dtype=tf.float32, shape=[84, 15, 1])

diag_mat_1 = sigma_1
diag_mat_2 = sigma_2

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
        w = tf.get_variable(name="w", shape = [22, output_size], dtype= tf.float64, initializer=tf.constant_initializer(0.01))
        #w = tf.Print(w,[w], message="my W-values:")

        b = tf.get_variable(name="b", shape = [output_size], dtype=tf.float64, initializer=tf.constant_initializer(0.01))
        res1 = tf.matmul(input_ , w)
        #b = tf.Print(b, [b], message="my B-values:"+scope)
        #if activation is None:
        #    res = tf.nn.xw_plus_b(input_, w, b)
        #else:
        #    res = activation(tf.nn.xw_plus_b(input_, w, b))

        #if dropout:
        #    res = tf.nn.dropout(res, rate = 0.2)
        res = tf.add(res1, b)
        return res, res1

y_s_enc = tf.placeholder(shape=[84,22], dtype=tf.float64)
enc_zeta_hidden, inter = fc_layer( y_s_enc, 15,  scope="zeta_hidden" )
                    
y_ = np.loadtxt("enc_y_s_1.txt")
z = np.zeros([22, 15])
z.fill(0.01)
check1 = np.matmul(y_, z)
print(y_[0])
print(check1[0])
'''
check1 = np.matmul(y_[1], z)
print(y_[1])
print(check1)
'''
'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())
enc_z, inter = sess.run([enc_zeta_hidden, inter], feed_dict={y_s_enc : y_})
print("enc_z", enc_z)
print("inter", inter)
'''
