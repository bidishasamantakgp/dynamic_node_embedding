import tensorflow as tf
import numpy as np 

data = np.reshape(np.arange(30), [1, 30])
x = tf.constant(data)
result = tf.cast(tf.gather(x, 0), dtype=tf.float32)
res1 = tf.gather(result,1)
sess = tf.Session()

print data
print sess.run(result), result.get_shape() 
print sess.run(res1), res1.get_shape()
