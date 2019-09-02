import tensorflow as tf
import numpy as np
import time
from datetime import datetime
from tensorflow.python.client import device_lib
def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']

#'''
with tf.device('/gpu:1'):
    # Matmul experiment
    '''
    a = tf.placeholder(dtype = tf.float32, shape = [10, 10])
    b = tf.placeholder(dtype = tf.float32, shape = [10,10])
    #a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    #b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    d = []
    
    for i in range(10):
        for j in range(10):
            c = []
            for k in range(10):
                c.append(a[i][j] * b[j][k])
            d.append(tf.reduce_sum(c))
    d = tf.stack(d)
    
    c = tf.matmul(a, b)
    '''
    #'''
    y = tf.placeholder(dtype = tf.float32, shape = [84,1])
    b = tf.placeholder(dtype = tf.float32, shape = [1, 5])
    adj = tf.placeholder(dtype = tf.float32, shape = [84, 84])

    e = tf.placeholder(dtype = tf.float32, shape = [1, 4])
    u = tf.cast(e[0][0], tf.int32)
    v = tf.cast(e[0][1], tf.int32)
    y_t = []
    h = tf.placeholder(dtype = tf.float32, shape = [1, 10])
    for i in range(84):
                    temp1 = tf.gather_nd(adj, (i, u))
                    temp2 = tf.gather(y, u)
                    temp3 = tf.gather_nd(adj, (i,v))
                    temp4 = tf.gather(y, v)
                    temp5 = tf.multiply(temp1, temp2)
                    temp6 = tf.multiply(temp3, temp4)
                    intermediate = tf.add_n([temp5, temp6])
                    # print "Debug intermediate size", intermediate.get_shape(), h_inter.get_shape()
                    # intermediate = tf.add_n(tf.multiply(tf.gather_nd(self.adj, (i, u)), tf.gather(y, u)), tf.multiply(tf.gather_nd(self.adj, (i,v)), tf.gather(y, v)))
                    first = tf.concat([[y[i]], [intermediate], h], axis=1)
                    second = tf.concat([tf.zeros([1, 2 * 1]), h], axis=1)
                    y_t.append(tf.concat([tf.add(first, second), b], axis = 1))
    
    c = tf.stack(y_t)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    #h = tf.placeholder(dtype = tf.float32, shape = [1])

# Runs the op.
summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
print("len", len(tf.get_default_graph().get_operations()))
t1 = time.time()
print(sess.run([c], feed_dict = {y:np.ones([84, 1]), b:np.zeros([1, 5]), h:np.ones([1, 10]), e:[[1,2,0,0]], adj:np.ones([84, 84])}))
t2 = time.time()

print("Time", t2 - t1)
#print get_available_gpus()

