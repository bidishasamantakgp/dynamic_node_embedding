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

sigma1 = tf.matrix_diag(sigma_1)
sigma2 = tf.matrix_diag(sigma_2)
sigma_2_inv = tf.matrix_diag(tf.divide(tf.ones([84, 15], tf.float32), diag_mat_2))
#sigma_2_inv = tf.Print(sigma_2_inv, [sigma_2_inv], message = "Sigma_2_inv")

sigma_2_inv_sigma_1 = tf.matmul(sigma_2_inv, tf.transpose(sigma1, [0, 2, 1]))
trace_val = tf.trace(sigma_2_inv_sigma_1)

mu_diff = tf.subtract(mu_2, mu_1)
sigma_mu_sigma = tf.reshape(tf.matmul(tf.matmul(tf.transpose(mu_diff,[0, 2, 1]), sigma_2_inv), mu_diff), [84])

# diag_mat_1 = tf.tensor_diag_part(sigma_1)
# diag_mat_2 = tf.tensor_diag_part(sigma_2)
# temp1 = tf.reduce_prod(tf.clip_by_value(diag_mat_1, clip_value_min=1e-09, clip_value_max=10), axis=1)
# temp2 = tf.reduce_prod(tf.clip_by_value(diag_mat_2, clip_value_min=1e-09, clip_value_max=10), axis=1)

# temp1 = tf.clip_by_value(tf.reduce_prod(diag_mat_1, axis=1), clip_value_min=1e-09, clip_value_max=1e10)
# temp2 = tf.clip_by_value(tf.reduce_prod(diag_mat_2, axis=1), clip_value_min=1e-09, clip_value_max=1e10)

clip1 = tf.clip_by_value(diag_mat_1, clip_value_min=1e-09, clip_value_max=1e10)
clip2 = tf.clip_by_value(diag_mat_2, clip_value_min=1e-09, clip_value_max=1e10)

log_det_1 = tf.reduce_sum(tf.log(tf.clip_by_value(diag_mat_1, clip_value_min=1e-09, clip_value_max=1e10)))
log_det_2 = tf.reduce_sum(tf.log(tf.clip_by_value(diag_mat_2, clip_value_min=1e-09, clip_value_max=1e10)))


#log_det_1 = tf.log(tf.maximum(temp1, tf.fill([84], tf.cast(1e-09, tf.float32))))
#log_det_2 = tf.log(tf.maximum(temp2, tf.fill([84], tf.cast(1e-09, tf.float32))))
det_val = tf.divide(log_det_2, log_det_1)
                    
#print("Debug size of everyone",get_shape(trace_val), get_shape(sigma_mu_sigma), get_shape(det_val) )
kl = tf.reduce_sum(0.5 * (trace_val + sigma_mu_sigma + det_val - k))

s1 = np.loadtxt("enc_z_sigma_debug_1.txt")
mu1 = np.reshape(np.loadtxt("enc_z_mu_debug_1.txt"), [84, 15, 1])
s2 = np.loadtxt("prior_z_sigma_debug_1.txt")
mu2 = np.reshape(np.loadtxt("prior_z_mu_debug_1.txt"), [84, 15, 1])
print("s1", s1[5])
print("s2", s2[5])
#print(KL(mu1, s1, mu2, s2))

sess = tf.Session()
kl, sigma_2_inv, sigma_2_inv_sigma_1, trace_val, sigma_mu_sigma, diag_mat_1, diag_mat_2, log_det_1, log_det_2, clip1, clip2 = sess.run([kl, sigma_2_inv, sigma_2_inv_sigma_1, trace_val, sigma_mu_sigma, diag_mat_1, diag_mat_2, log_det_1, log_det_2, clip1, clip2], feed_dict = {sigma_1:s1, sigma_2:s2, mu_1:mu1, mu_2:mu2})
print("KL", kl)
'''
print("Sigma_2_inv", sigma_2_inv)
print("Sigma_2_inv_sigma_1", sigma_2_inv_sigma_1)
print("trace_val", trace_val)
print("sigma_mu_sigma", sigma_mu_sigma)
'''
print("diag_mat_1", diag_mat_1[5])
print("diag_mat_2", diag_mat_2[5])
for i in range(84):
    print("clip_1", clip1[i])
    print("clip_2", clip2[i])

print("log_det_1", log_det_1)
print("log_det_2", log_det_2)

#print("temp_1", temp1[5])
#print("temp_2", temp2[5])
#'''
