import tensorflow as tf                
import numpy as np

def KL_tf(sigma_1, sigma_2, mu_1, mu_2):
            k = tf.fill([84], tf.cast(15, tf.float32))

            with tf.variable_scope("kl_gaussisan"):

                #diag_mat_1 = tf.diag_part(sigma_1)
                #diag_mat_2 = tf.diag_part(sigma_2)

                diag_mat_1 = sigma_1
                diag_mat_2 = sigma_2

                sigma_1 = tf.matrix_diag(sigma_1)
                sigma_2 = tf.matrix_diag(sigma_2)
                sigma_2_inv = tf.matrix_diag(tf.divide(tf.ones([84, 15]), diag_mat_2))
                #sigma_2_inv = tf.Print(sigma_2_inv, [sigma_2_inv], message = "Sigma_2_inv")

                sigma_2_inv_sigma_1 = tf.matmul(sigma_2_inv, tf.transpose(sigma_1, [0, 2, 1]))
                trace_val = tf.trace(sigma_2_inv_sigma_1)

                mu_diff = tf.subtract(mu_2, mu_1)
                sigma_mu_sigma = tf.reshape(tf.matmul(tf.matmul(tf.transpose(mu_diff,[0, 2, 1]), sigma_2_inv), mu_diff), [84])

                #diag_mat_1 = tf.tensor_diag_part(sigma_1)
                #diag_mat_2 = tf.tensor_diag_part(sigma_2)
                log_det_1 = tf.log(tf.maximum(tf.reduce_prod(diag_mat_1, axis=1), tf.fill([84], 1e-09)))
                log_det_2 = tf.log(tf.maximum(tf.reduce_prod(diag_mat_2, axis=1), tf.fill([84], 1e-09)))
                det_val = tf.divide(log_det_2, log_det_1)
                    
                print("Debug size of everyone",get_shape(trace_val), get_shape(sigma_mu_sigma), get_shape(det_val) )
                return tf.reduce_sum(0.5 * (trace_val + sigma_mu_sigma + det_val - k))

def KL(mu_1, sigma_1, mu_2, sigma_2):

                diag_mat_1 = sigma_1
                diag_mat_2 = sigma_2
                sigma_2_inv = np.divide(np.ones([84, 15]), sigma_2)
                sigma_1_list = []
                sigma_2_list = []
                for i in range(84):
                    sigma_1_list.append(np.diag(sigma_1[i]))
                    sigma_2_list.append(np.diag(sigma_2_inv[i]))

                s1 = np.stack(sigma_1_list)
                s2 = np.stack(sigma_2_list)
                print("S2", s2)
                #sigma_2_inv = np.divide(np.ones([84, 15, 15]), s2)
                #sigma_2_inv = tf.Print(sigma_2_inv, [sigma_2_inv], message = "Sigma_2_inv")
                print("inv", sigma_2_inv)
                sigma_2_inv_sigma_1 = np.matmul(s2, np.transpose(s1, [0, 2, 1]))
                print(sigma_2_inv_sigma_1.shape)
                trace_val_l = []
                for i in range(84):
                    trace_val_l.append(np.trace(sigma_2_inv_sigma_1[i]))
                trace_val = np.stack(trace_val_l)
                mu_diff = np.subtract(mu_2, mu_1)
                print(mu_diff.shape)
                sigma_mu_sigma = np.reshape(np.matmul(np.matmul(np.transpose(mu_diff,[0, 2, 1]), s2), mu_diff), [84])

                #diag_mat_1 = tf.tensor_diag_part(sigma_1)
                #diag_mat_2 = tf.tensor_diag_part(sigma_2)
                z = np.zeros([84])
                z.fill(1e-09)
                log_det_1 = np.log(np.maximum(np.prod(diag_mat_1, axis=1), z))
                log_det_2 = np.log(np.maximum(np.prod(diag_mat_2, axis=1), z))
                det_val = np.divide(log_det_2, log_det_1)
                k = np.zeros([84])
                k.fill(15.0)
                print("Size", trace_val.shape, sigma_mu_sigma.shape, det_val.shape, k.shape)
                return np.sum(0.5 * (trace_val + sigma_mu_sigma + det_val - k))

s1 = np.loadtxt("enc_z_sigma_debug_0.txt")
mu1 = np.reshape(np.loadtxt("enc_z_mu_debug_0.txt"), [84, 15, 1])
s2 = np.loadtxt("prior_z_sigma_debug_0.txt")
mu2 = np.reshape(np.loadtxt("prior_z_mu_debug_0.txt"), [84, 15, 1])
print(KL(mu1, s1, mu2, s2))

self.sess = tf.Session()


