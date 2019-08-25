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
    #b1 = tf.matmul(tf.ones([84,1]), b)
    #c = tf.concat([a,b1], axis = 1)
    #'''
    #c_s = []
    #for i in range(84): 
    #    c_s.append(tf.concat([[a[i]], b], axis = 1))
    #c = tf.stack(c_s)
    # Creates a session with log_device_placement set to True.
    #config = tf.ConfigProto(allow_soft_placement = True)
    #sess = tf.Session()
    h = tf.placeholder(dtype = tf.float32, shape = [1, 10])
    temp_u = tf.reshape(tf.one_hot(indices=u, depth=84), [1, 84])
    temp_v = tf.reshape(tf.one_hot(indices=v, depth=84), [1, 84])
    
    y_u = tf.matmul(temp_u, y)
    y_v = tf.matmul(temp_v, y)

    u_append = tf.matmul(tf.matmul(adj , tf.transpose(temp_u)), y_u)
    v_append = tf.matmul(tf.matmul(adj , tf.transpose(temp_v)), y_v)

    append_val = tf.add(u_append, v_append)
 
    h_inter = tf.concat([b, h], axis=1)
    
    h_outscale = tf.matmul(tf.ones([84, 1]), h_inter)

    
    #c_h = []
    
    #for i in range(84):
    #    tf.gather()
    c = tf.concat([y, append_val, h_outscale], axis = 1 )

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    #h = tf.placeholder(dtype = tf.float32, shape = [1])

# Runs the op.
summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
print("len", len(tf.get_default_graph().get_operations()))
for i in tf.get_default_graph().get_operations():
    print("OP: ", i)
t1 = time.time()
print(sess.run([c], feed_dict = {y:np.ones([84, 1]), b:np.zeros([1, 5]), h:np.ones([1, 10]), e:[[1,2,0,0]], adj:np.ones([84, 84])}))
t2 = time.time()

print("Time", t2 - t1)
#print get_available_gpus()

