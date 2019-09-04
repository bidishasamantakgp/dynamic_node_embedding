import tensorflow as tf
from utils import *
from cell_matrix_new import MPPCell
import time
import copy
from datetime import datetime
import numpy as np
import os
class MPPModel():
    def __init__(self, args, sample=False):

        self.n = args.n

        def tf_likelihood(y, l_a, l_c):
            with tf.variable_scope('likelihood'):
                u = y[0][0]
                v = y[0][1]
                m = y[0][3]
                t = y[0][2]

                index = self.n * (u-1) + v
                l_a_occured = tf.cast([1 - m], tf.float32) * tf.gather(l_a, index)
                l_c_occured = tf.cast([m], tf.float32) * tf.gather(l_c, index)

                l_a_comp = tf.reduce_sum(l_a) - l_a_occured
                l_c_comp = tf.reduce_sum(l_c) - l_c_occured

                association_loss =  tf.cast([1 - m], tf.float32) * tf.subtract(tf.log(tf.maximum(l_a_occured, 1e-09)), tf.add(l_a_comp, l_c_comp))
                communication_loss = tf.cast([m], tf.float32) * tf.subtract(tf.log(tf.maximum(l_c_occured, 1e-09)), tf.add(l_a_comp, l_c_comp))
                ll = tf.reduce_mean(association_loss + communication_loss)

                return (association_loss, communication_loss, ll)

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2, _str="zeta"):

            k = tf.fill([self.n], tf.cast(args.z_dim, tf.float32))

            with tf.variable_scope("kl_gaussisan"):
        
                #diag_mat_1 = tf.diag_part(sigma_1)
                #diag_mat_2 = tf.diag_part(sigma_2)
                
                diag_mat_1 = sigma_1
                diag_mat_2 = sigma_2

                sigma_1 = tf.matrix_diag(sigma_1)
                sigma_2 = tf.matrix_diag(sigma_2)
                sigma_2_inv = tf.matrix_diag(tf.divide(tf.ones([self.n, args.z_dim]), diag_mat_2))
                #sigma_2_inv = tf.Print(sigma_2_inv, [sigma_2_inv], message = "Sigma_2_inv")

                sigma_2_inv_sigma_1 = tf.matmul(sigma_2_inv, tf.transpose(sigma_1, [0, 2, 1]))
                trace_val = tf.trace(sigma_2_inv_sigma_1)

                mu_diff = tf.subtract(mu_2, mu_1)
                sigma_mu_sigma = tf.reshape(tf.matmul(tf.matmul(tf.transpose(mu_diff,[0, 2, 1]), sigma_2_inv), mu_diff), [self.n])
                
                #diag_mat_1 = tf.tensor_diag_part(sigma_1)
                #diag_mat_2 = tf.tensor_diag_part(sigma_2)
                #log_det_1 = tf.log(tf.clip_by_value(tf.reduce_prod(diag_mat_1, axis=1), clip_value_min=1e-09, clip_value_max=1e10))
                #log_det_2 = tf.log(tf.clip_by_value(tf.reduce_prod(diag_mat_2, axis=1), clip_value_min=1e-09, clip_value_max=1e10))
                
                log_det_1 = tf.reduce_sum(tf.log(tf.clip_by_value(diag_mat_1, clip_value_min=1e-09, clip_value_max=1e10)))
                log_det_2 = tf.reduce_sum(tf.log(tf.clip_by_value(diag_mat_2, clip_value_min=1e-09, clip_value_max=1e10)))

                #log_det_1 = tf.log()
                #log_det_2 = tf.log(tf.maximum(tf.reduce_prod(diag_mat_2, axis=1), tf.fill([self.n], 1e-09)))
                det_val = tf.subtract(log_det_2, log_det_1)
                
                k = tf.fill([self.n], tf.cast(args.z_dim, tf.float32))
                #det_val = tf.Print(det_val, [det_val], message='Detval:')
                print("Debug size of everyone",get_shape(trace_val), get_shape(sigma_mu_sigma), get_shape(det_val) )
                return tf.reduce_sum(0.5 * (trace_val + sigma_mu_sigma + det_val - k))


        def get_lossfunc(l_c, l_a, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma, y):
            loss = 0.0
            #y = y[0]
            y = tf.transpose(y, [1,0,2] )

            self.kl_loss_zeta_list = []
            self.kl_loss_z_list = []
            self.ll_list = []
            self.a_l_list = []
            self.c_l_list = []
            
            for i in range(args.seq_length):
                    #print "Debug i", i, l_c[i], l_a[i]
                    #with tf.device('/cpu:0'):
                    print("Zeta KL")
                    kl_loss_zeta = tf_kl_gaussgauss(enc_zeta_mu[i], enc_zeta_sigma[i], prior_zeta_mu[i], prior_zeta_sigma[i])
                    print("Z KL")
                    kl_loss_z = tf_kl_gaussgauss(enc_z_mu[i], enc_z_sigma[i], prior_z_mu[i], prior_z_sigma[i], "Z")
                    self.kl_loss_zeta_list.append(kl_loss_zeta)
                    self.kl_loss_z_list.append(kl_loss_z)
                    a_l, c_l, likelihood_loss = tf_likelihood(y[i],l_a[i], l_c[i])
                    self.ll_list.append(likelihood_loss)
                    self.a_l_list.append(a_l)
                    self.c_l_list.append(c_l)
                    #with tf.device('/cpu:0'):
                    loss += tf.reduce_mean(kl_loss_zeta + kl_loss_z - likelihood_loss)
            return (loss/args.seq_length)

        self.args = args

        if sample:
            args.batch_size = 1
            args.seq_length = 1

        # cell = MPPCell(self.adj, self.features, args.sample, self.eps, args.k, args.h_dim, args.n_c, args.z_dim)
        # MPPCell(args.chunk_samples, args.rnn_size, args.latent_size)
        # self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length, 4], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length, 4], name='target_data')
        #self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.seq_length, self.n, self.n], name='target_data')
        #self.m = tf.placeholder(dtype=tf.float32, shape=[args.seq_length, 1], name='type_messege')

        self.features = tf.placeholder(dtype=tf.float32, shape=[args.n, args.d_dim], name='features')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[args.n, args.z_dim, 1], name='eps')
        self.adj = tf.placeholder(dtype=tf.float32, shape=[args.seq_length, args.n, args.n], name='adj')
        self.adj_prev = tf.placeholder(dtype=tf.float32, shape=[args.seq_length, args.n, args.n], name='prev')
        self.B_old = tf.placeholder(dtype=tf.float32, shape=[args.n_c, args.n_c], name='B')
        self.r_old = tf.placeholder(dtype=tf.float32, shape=[args.n, args.n], name='R')
        self.time_cur = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length, 1], name='T')

        #Trial
        self.initial_state_t = tf.placeholder(dtype = tf.float32, shape = [1, args.h_dim], name = "s_t")
        self.initial_state_s = tf.placeholder(dtype = tf.float32, shape = [1, args.h_dim], name = "s_c")
        #self.event_indicator = tf.placeholder(dtype = tf.int32, shape = [args.n], name = "indicator")
        #with tf.device('/device:GPU:1'):
        cell = MPPCell(args, self.features, self.eps)

        self.cell = cell
        #debug_state_size = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        #print "Debug state size", len(debug_state_size), debug_state_size
        list_debug = (self.initial_state_t, self.initial_state_s)
        #rint "Debug input", tf.stack(list_debug), list_debug
        #[self.initial_state_t, self.initial_state_s] = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
            inputs = tf.reshape(inputs, [-1, 4]) # (n_steps*batch_size, n_input)
            

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
        inputs = tf.split(axis=0, num_or_size_splits=args.seq_length, value=inputs) # n_steps * (batch_size, n_hidden)
       # print "Debug input shpae", self.input_data
        #Get vrnn cell output
        #outputs, last_state = tf.nn.dynamic_rnn(cell=self.cell, dtype=tf.float32, inputs=self.input_data, initial_state=list_debug)

        state = (self.initial_state_t, self.initial_state_s)
        outputs = []
        B = self.B_old
        r = self.r_old
        with tf.variable_scope("RNN"):
            for time_step in range(args.seq_length):
                    tf.get_variable_scope().reuse_variables()
                    (cell_output, state_t, state_s) = self.cell.new_call(self.input_data[:, time_step, :], self.adj[time_step], self.adj_prev[time_step], self.time_cur[:,time_step, :], state, B, r)
                    state = (state_t, state_s)
                    B = cell_output[-4]
                    r = cell_output[-3]
                    #print "debug output", cell_output
                    outputs.append(cell_output)

        outputs_reshape = []
        names = ["h_inter", "y_current", "y_s", "y_s_enc", "y_t","hidded", "l_c", "l_a", "enc_zeta_mu", "enc_z_mu", "enc_z_sigma", "enc_zeta_sigma", "prior_z_mu", "prior_zeta_mu", "prior_z_sigma", "prior_zeta_sigma", "B", "r", "P", "C"]

        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                #x = tf.transpose(x,[1,0,2])
                #x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)
        #h_inter_enc, y_current, y_s_stack, y_s_enc_stack, y_t_enc_stack, enc_zeta_hidden, l_c, l_a, enc_zeta_mu, enc_z_mu, enc_zeta_sigma, enc_z_sigma, prior_zeta_mu, prior_z_mu, prior_zeta_sigma, prior_z_sigma, B, r
        self.h_inter_enc, self.y_current, self.y, self.y_s_enc, self.y_t, self.hidden, l_c, l_a, self.enc_zeta_mu, self.enc_z_mu, self.enc_zeta_sigma, self.enc_z_sigma, self.prior_zeta_mu, self.prior_z_mu, self.prior_zeta_sigma, self.prior_z_sigma, self.B_new, self.r_new, self.P, self.C = outputs_reshape
        self.final_state_t, self.final_state_s = state
        #self.mu = dec_mu
        #self.sigma = dec_sigma
        #print "Debug size before the lossfunc", enc_zeta_mu
        #(l_c, l_a, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma, y
        lossfunc = get_lossfunc(l_c, l_a, self.enc_zeta_mu, self.enc_z_mu, self.enc_z_sigma, self.enc_zeta_sigma, self.prior_z_mu, self.prior_zeta_mu, self.prior_z_sigma, self.prior_zeta_sigma, self.target_data)

        self.l_c = l_c
        self.l_a = l_a

        with tf.variable_scope('cost'):
            self.cost = lossfunc

        #print 'cost', self.cost
        #print 'lambda_communication', l_c
        #print 'lambda_association', l_a

        #tf.summary.tensor_summary('cost', self.cost)
        #tf.summary.tensor_summary('lambda_communication', l_c)
        #tf.summary.tensor_summary('lambda_association', l_a)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        print_vars("trainable_variables")
        #for t in tvars:
        #    print "trainable vars", t.name
        t1 = time.time()
        #with tf.device('/gpu:7'):
        grads = tf.gradients(self.cost, tvars)
        t2 = time.time()
        print("After grad:", t2 - t1)
        #with tf.device('/gpu:7'):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.sess = tf.Session()

    def sample(self):
        #TBD
        return ""
    def initialize(self):
        #logger.info("Initialization of parameters")
        #self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())
    
    def restore(self, savedir):
        #saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(savedir)
        if ckpt == None or ckpt.model_checkpoint_path == None:
            self.initialize()
        else:    
            print("Load the model from {}".format(ckpt.model_checkpoint_path))
            #saver.restore(self.sess, ckpt.model_checkpoint_path)
    
    def train(self, args, samples):

            dirname = args.out_dir
            #print "Time size of graph", len(tf.get_default_graph().get_operations())

            ckpt = tf.train.get_checkpoint_state(dirname)

            n_batches = len(samples)
            sample_train, sample_test = create_samples(args)
            ##config = tf.ConfigProto(
            ##allow_soft_placement=True,
            ##log_device_placement=False
            ##)
            ##config.gpu_options.allow_growth = True
    
            ##with tf.Graph().as_default(), tf.Session(config=config) as sess:
            ##with tf.Session() as sess:
            sess = self.sess
            summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
            #check = tf.add_check_numerics_ops()
            #merged = tf.summary.merge_all()
            #tf.global_variables_initializer().run()
            ckpt = tf.train.get_checkpoint_state(args.out_dir)
            saver = tf.train.Saver(tf.global_variables())


            #initial_state_t = np.zeros([1, args.h_dim])
            #initial_state_s = np.zeros([1, args.h_dim])
            features = get_one_hot_features(self.n)

            eps = np.random.randn(args.n, args.z_dim, 1)
            #B_old_1 = np.zeros([args.n_c, args.n_c])
            #r_old_1 = np.zeros([args.n, args.n])

            adj_old = starting_adj(args, samples)
            
            '''
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loaded model")
            '''
            start = time.time()
            sess.graph.finalize()
            #tf.reset_default_graph()
            
            for e in range(args.num_epochs):
                print("Size of graph", len(tf.get_default_graph().get_operations()))
                #for op in tf.get_default_graph().get_operations():
                #    print("OP: ", op)
                #sess.run(tf.assign(self.lr, args.learning_rate * (args.decay_rate ** e)))
                adj_list = []
                adj_list_prev = []
                B_old = np.zeros([args.n_c, args.n_c])
                r_old = np.zeros([args.n, args.n])
                initial_state_t = np.zeros([1, args.h_dim])
                initial_state_s = np.zeros([1, args.h_dim])
                for b in range(len(samples)):
                    #np.savetxt("B_old.txt"+str(b), B_old)
                    #np.savetxt("R_old.txt"+str(b), r_old)

                    x, y = next_batch(args, samples, b)
                    #x = np.reshape(x, [])
                    time_next = extract_time(args, y)
                    '''
                    if len(adj_list_prev) > 0:
                         adj_list_prev[0] = copy.copy(adj_list[-1])
                    adj_list = get_adjacency_list(x, adj_old, args.n)

                    if len(adj_list_prev) > 0:
                         adj_list_prev[1:] = copy.copy(adj_list[:-1])
                    else:
                         adj_list_prev = [np.zeros((args.n, args.n))]

                    '''
                    if len(adj_list_prev) > 0:
                        adj_list_prev[0] = copy.copy(adj_list[-1])
                    adj_list = get_adjacency_list(x, adj_old, args.n)

                    if len(adj_list_prev) > 0:
                        adj_list_prev[1:] = copy.copy(adj_list[:-1])
                    else:
                        #print("Inside else")
                        adj_list_prev = [np.zeros((args.n, args.n))]
                        #print("Debug", len(adj_list_prev))
                        adj_list_prev.extend(adj_list[:-1])

                    ''' 
                    if len(adj_list_prev) > 0:
                        adj_list_prev[0] = adj_list[-1]
                    adj_list = get_adjacency_list(x, adj_old, self.n)

                    if len(adj_list_prev) > 0:
                        adj_list_prev[1:] = adj_list[:-1]
                    else:
                        adj_list_prev = [np.zeros((self.n, self.n))].append(adj_list[:-1])
                    '''

                    #print("Debug adj_list_prev:", adj_list_prev)
                    #for i_index in range(84):
                    #    print(adj_list_prev[0][i_index])

                    #print("Debug adj_list:")
                    #for i_index in range(84):
                    #    print(adj_list[0][i_index])

                    adj_old = adj_list[-1]

                    feed = {self.initial_state_s:initial_state_s, self.initial_state_t:initial_state_t,\
                    self.input_data: x, self.target_data: y, self.features: features, self.eps:eps, \
                    self.B_old: B_old, self.r_old: r_old, self.adj: adj_list, self.adj_prev: adj_list_prev, \
                    self.time_cur: time_next}
                    P, h_inter_enc, y_t_enc, y_s, y_s_enc, y_current, hidden, a_l, c_l, l_a, l_c, ll_list, kl_loss_zeta_list, kl_loss_z_list, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma = sess.run(
                    [self.P, self.h_inter_enc, self.y_t, self.y, self.y_s_enc, self.y_current, self.hidden, self.a_l_list, self.c_l_list, self.l_a, self.l_c, self.ll_list, self.kl_loss_zeta_list, self.kl_loss_z_list ,self.enc_zeta_mu, self.enc_z_mu, self.enc_z_sigma, self.enc_zeta_sigma, self.prior_z_mu, self.prior_zeta_mu, self.prior_z_sigma, self.prior_zeta_sigma], feed)

                    #print("Debug Y", y, "\nY_s", y_s, "\ny_s_enc", y_s_enc)
                    #print("Debug enc_zeta_sigma",  enc_zeta_sigma)
                    #print("Debug enc_z_sigma:")
                    #np.savetxt("enc_z_sigma_debug_0.txt", np.array(enc_z_sigma[0]))
                    #np.savetxt("enc_z_sigma_debug_1.txt", np.array(enc_z_sigma[1]))
                    #np.savetxt("enc_z_mu_debug_0.txt", np.reshape(enc_z_mu[0], [84, 15]))
                    #np.savetxt("enc_z_mu_debug_1.txt", np.reshape(enc_z_mu[1], [84, 15]))

                    
                    #np.savetxt("eps.txt", np.reshape(eps, [84, 15]))      
                    #for j1 in range(10):
                    #    print("j1", j1)
                    #    np.savetxt("enc_z_sigma_debug_"+str(j1)+".txt", np.array(enc_z_sigma[j1]))
                    #    #np.savetxt("enc_z_mu_"+str(j1)+".txt", np.reshape(enc_z_mu[j1], [84,15]))
                    #    for i in range(84):
                    #        print(enc_z_sigma[j1][i])
                    
                    #print("Debug prior z_sigma:")
                    #np.savetxt("prior_z_sigma_debug_0.txt", np.array(prior_z_sigma[0]))
                    #np.savetxt("prior_z_sigma_debug_1.txt", np.array(prior_z_sigma[1]))
                    #np.savetxt("prior_z_mu_debug_0.txt", np.reshape(prior_z_mu[0], [84, 15]))
                    #np.savetxt("prior_z_mu_debug_1.txt", np.reshape(prior_z_mu[1], [84, 15]))

                    #for j1 in range(10):
                    #    #np.savetxt("prior_z_sigma_"+str(j1)+".txt", prior_z_sigma[j1])
                    #    #np.savetxt("prior_z_mu_"+str(j1)+".txt", np.reshape(prior_z_mu[j1], [84, 15]))
                    #    print(j1)
                    #    for i in range(84):
                    #        print(prior_z_sigma[j1][i])

                    print("Debug loss KL_Z:",kl_loss_z_list)
                    print("Debug loss KL_zeta:", kl_loss_zeta_list)
                    print("Debug loss LL_loss:", ll_list)
                    
                    #np.savetxt("la_0.txt", np.reshape(l_a[0], [84, 84]))
                    #np.savetxt("lc_0.txt", np.reshape(l_c[0], [84, 84]))
                    train_loss, cr, initial_state_s, initial_state_t, B_old_1, r_old_1, C = sess.run(
                            [self.cost, self.train_op, self.final_state_s, self.final_state_t, self.B_new, self.r_new, self.C], feed)
                    #summary_writer.add_summary(summary, e * n_batches + b)
                    '''
                    np.savetxt("B.txt", B_old_1[7])
                    np.savetxt("R.txt", r_old_1[7])
                    np.savetxt("P.txt", np.reshape(P[7], [84, -1]))
                    np.savetxt("C.txt", C[7])
                    np.savetxt("Y_s.txt", y_s_enc[7])
                    np.savetxt("Y_t.txt", y_t_enc[7])
                    np.savetxt("B1.txt"+str(b), B_old_1[-1])
                    np.savetxt("R1.txt"+str(b), r_old_1[-1])
                    #np.savetxt("P1.txt", np.reshape(P[1], [84, -1]))
                    #np.savetxt("C1.txt", C[1])
                    np.savetxt("B2.txt", B_old[2])
                    np.savetxt("R2.txt", r_old[2])
                    np.savetxt("P2.txt", np.reshape(P[2], [84, -1]))
                    np.savetxt("C2.txt", C[2])

                    np.savetxt("B2.txt", B_old[2])
                    np.savetxt("R2.txt", r_old[2])
                    np.savetxt("P2.txt", np.reshape(P[2], [84, -1]))
                    np.savetxt("C2.txt", C[2])

                    np.savetxt("B_1.txt", B_old[-1])
                    np.savetxt("R_1.txt", r_old[-1])
                    np.savetxt("P_1.txt", np.reshape(P[-1], [84, -1]))
                    np.savetxt("C_1.txt", C[-1])
                    '''
                    if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                        checkpoint_path = os.path.join(dirname, 'model.ckpt')
                        #saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                        print("model saved to {}".format(checkpoint_path))
                    end = time.time()
                    '''
                    '''
                    print("{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}" \
                        .format(e * n_batches + b,
                                args.num_epochs * n_batches,
                                e, args.seq_length * train_loss, end - start))
                    start = time.time()
                    B_old = np.reshape(B_old_1[-1], [args.n_c, args.n_c])
                    r_old = np.reshape(r_old_1[-1], [args.n, args.n])

                    initial_state_s = [initial_state_s[-1]]
                    initial_state_t = [initial_state_t[-1]]
            #tf.reset_default_graph()


    def predict_association(self, test_sample, adj):
        for b in range(len(samples)):
            x, y = next_batch(args, test_sample, b)
            feed = {self.input_data: x, self.target_data: y, self.adj: adj}
            l_c, l_a = self.run([self.l_c, self.l_a], feed)
            x = y
            feed = {self.input_data: x, self.target_data: y, self.adj: adj}
            l_c_n, l_a_n = self.run([self.l_c, self.l_a], feed)

            for i in range(len(l_c)):
                u = x[0][0]
                v = x[0][1]
                #t = x[0][2]
                #type_m = x[0][3]
                #l_a_n[i][u] l_a[i][u]



    def predict_fututure(l_a, l_c, ):
        return 0

