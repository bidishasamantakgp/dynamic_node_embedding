import tensorflow as tf
from utils import *
from cell_matrix import MPPCell
import time
import copy
from datetime import datetime
import numpy as np
import os
class MPPModel():
    def __init__(self, args, sample=False):

        self.n = args.n
        self.lr = args.learning_rate
        def tf_likelihood(y, l_a, l_c):
            with tf.variable_scope('likelihood'):
                u = y[0][0]
                v = y[0][1]
                m = y[0][3]
                t = y[0][2]
                l_a_r = tf.reshape(l_a, [self.n * self.n])
                l_c_r = tf.reshape(l_c, [self.n * self.n])
                
                last_time_sr = tf.sqare(self.last_time)
                current_time_sr = tf.square(tf.fill(self.))
                index = self.n * (u) + v
                l_a_occured = tf.cast([1 - m], tf.float32) * tf.gather(l_a_r, index)
                l_c_occured = tf.cast([m], tf.float32) * tf.gather(l_c_r, index)

                l_a_comp = tf.reduce_sum(l_a) - l_a_occured
                l_c_comp = tf.reduce_sum(l_c) - l_c_occured

                association_loss =  tf.cast([1 - m], tf.float32) * tf.subtract(tf.log(tf.maximum(l_a_occured, 1e-09)), tf.add(l_a_comp, l_c_comp))
                communication_loss = tf.cast([m], tf.float32) * tf.subtract(tf.log(tf.maximum(l_c_occured, 1e-09)), tf.add(l_a_comp, l_c_comp))
                ll = tf.maximum(-1e14, tf.reduce_mean(association_loss + communication_loss))

                return (association_loss, communication_loss, ll)


        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2, _str="zeta"):
                #mu_1 = tf.reshape(mu_1, [self.n, -1])
                #mu_2 = tf.reshape(mu_2, [self.n, -1])

                return tf.reduce_mean(tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1.0, sigma_2),name='log_sigma_2')
                  - 2 * tf.log(tf.maximum(1.0, sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1.0, (tf.square(sigma_2))) - 1
                ), 1))

                #return tf.reduce_sum(0.5 * (
                #    2 * tf.log(tf.maximum(1e-9,sigma_2),name='log_sigma_2') 
                #  - 2 * tf.log(tf.maximum(1e-9,sigma_1),name='log_sigma_1')
                #  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
                #), 1)


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
                    kl_loss_zeta = tf.minimum(1e14, tf_kl_gaussgauss(enc_zeta_mu[i], enc_zeta_sigma[i], prior_zeta_mu[i], prior_zeta_sigma[i]))
                    print("Z KL")
                    kl_loss_z = tf.minimum(1e14 , tf_kl_gaussgauss(enc_z_mu[i], enc_z_sigma[i], prior_z_mu[i], prior_z_sigma[i], "Z"))
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
        self.initial_state = tf.placeholder(dtype = tf.float32, shape = [1, 2 * args.h_dim], name = "s")
        

        #with tf.device('/device:GPU:1'):
        cell = MPPCell(args, self.features, self.eps)

        self.cell = cell

        state = (self.initial_state, tf.zeros([1, args.h_dim]))
        #(self.initial_state_t, self.initial_state_s)
        outputs = []
        B = self.B_old
        r = self.r_old
        self.state_list = []
        with tf.variable_scope("RNN"):
            for time_step in range(args.seq_length):
                    tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = self.cell.new_call(self.target_data[:, time_step, :], self.input_data[:, time_step, :], self.adj[time_step], self.adj_prev[time_step], state, B, r)
                    #state = (state_t, state_s)
                    B = cell_output[-2]
                    r = cell_output[-1]
                    outputs.append(cell_output)
                    self.state_list.append(state[0])

        outputs_reshape = []
        #names = ["h_inter", "y_current", "y_s", "y_s_enc", "y_t","hidded", "l_c", "l_a", "enc_zeta_mu", "enc_z_mu", "enc_z_sigma", "enc_zeta_sigma", "prior_z_mu", "prior_zeta_mu", "prior_z_sigma", "prior_zeta_sigma", "B", "r", "P", "C", "B_h"]

        names = ["h_inter", "y_current", "y_s", "y_s_enc", "y_t","hidded", "l_c", "l_a", "enc_zeta_mu", "enc_z_mu", "enc_z_sigma", "enc_zeta_sigma", "prior_z_mu", "prior_zeta_mu", "prior_z_sigma", "prior_zeta_sigma", "B", "r"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                outputs_reshape.append(x)
        #self.h_inter_enc, self.y_current, self.y, self.y_s_enc, self.y_t, self.hidden, l_c, l_a, self.enc_zeta_mu, self.enc_z_mu, self.enc_zeta_sigma, self.enc_z_sigma, self.prior_zeta_mu, self.prior_z_mu, self.prior_zeta_sigma, self.prior_z_sigma, self.B_new, self.r_new, self.P, self.C, self.B_hidden = outputs_reshape
        self.h_inter_enc, self.y_current, self.y, self.y_s_enc, self.y_t, self.hidden, l_c, l_a, self.enc_zeta_mu, self.enc_z_mu, self.enc_zeta_sigma, self.enc_z_sigma, self.prior_zeta_mu, self.prior_z_mu, self.prior_zeta_sigma, self.prior_z_sigma, self.B_new, self.r_new = outputs_reshape
        self.final_state = state
        lossfunc = get_lossfunc(l_c, l_a, self.enc_zeta_mu, self.enc_z_mu, self.enc_z_sigma, self.enc_zeta_sigma, self.prior_z_mu, self.prior_zeta_mu, self.prior_z_sigma, self.prior_zeta_sigma, self.target_data)

        self.l_c = l_c
        self.l_a = l_a

        with tf.variable_scope('cost'):
            self.cost = lossfunc

        tvars = tf.trainable_variables()
        print_vars("trainable_variables")
        print("Len trainable", len(tvars))
        #for t in tvars:
        #    print "trainable vars", t.name
        t1 = time.time()
        #with tf.device('/gpu:7'):
        self.grads = tf.gradients(self.cost, tvars)
        t2 = time.time()
        print("After grad:", t2 - t1)
        #with tf.device('/gpu:7'):
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(self.grads, tvars))
        self.sess = tf.Session()

    def sample(self):
        #TBD
        return ""
    def initialize(self):
        #logger.info("Initialization of parameters")
        #self.sess.run(tf.initialize_all_variables())
        self.sess.run(tf.global_variables_initializer())
    
    def restore(self, savedir):
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(savedir)
        if ckpt == None or ckpt.model_checkpoint_path == None:
            self.initialize()
        else:    
            print("Load the model from {}".format(ckpt.model_checkpoint_path))
            saver.restore(self.sess, ckpt.model_checkpoint_path)
    
    def train(self, args, samples):
            n_batches = len(samples)
            sample_train, sample_test = create_samples(args)
            sess = self.sess
            summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
            ckpt = tf.train.get_checkpoint_state(args.out_dir)
            saver = tf.train.Saver(tf.global_variables())
            dirname = args.out_dir
            
            features = get_one_hot_features(self.n)
            eps = np.random.randn(args.n, args.z_dim, 1)
            np.savetxt(dirname +"/initial_eps.txt", np.reshape(eps, [args.n, -1]))
            adj_old = starting_adj(args, args.start+1, samples)
            adj_old_prev = starting_adj(args, args.start, samples)
            
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print("Loaded model")
            
            start = time.time()
            sess.graph.finalize()
            #tf.reset_default_graph()
            tvars = tf.trainable_variables()            
            initial_state_rand = np.random.rand(1, 2 * args.h_dim)
            np.savetxt(dirname+"/initial_state.txt", initial_state_rand)
            for e in range(args.num_epochs):
                print("Size of graph", len(tf.get_default_graph().get_operations()))
                adj_list = []
                adj_list_prev = []
                B_old = np.zeros([args.n_c, args.n_c])
                B_old.fill(0.5)
                r_old = np.zeros([args.n, args.n])
                #initial_state = np.random.rand(1, 2 * args.h_dim)
                initial_state = initial_state_rand
                for b in range(len(samples)):
                    x, y = next_batch(args, samples, b)
                    # x = np.reshape(x, [])
                    time_next = extract_time(args, y)
                
                    adj_list = get_adjacency_list(y, adj_old, args.n)
                    adj_list_prev = get_adjacency_list(x, adj_old_prev, args.n)
                    '''
                    if len(adj_list_prev) > 0:
                        adj_list_prev[0] = copy.copy(adj_list[-1])
                    adj_list = get_adjacency_list(y, adj_old, args.n)

                    if len(adj_list_prev) > 0:
                        adj_list_prev[1:] = copy.copy(adj_list[:-1])
                    else:
                        # print("Inside else")
                        adj_list_prev = [np.zeros((args.n, args.n))]
                        # print("Debug", len(adj_list_prev))
                        adj_list_prev.extend(adj_list[:-1])
                    '''
                    adj_old = adj_list[-1]
                    adj_old_prev = adj_list_prev[-1]
                    feed = {self.initial_state:initial_state, \
                    self.input_data: x, self.target_data: y, self.features: features, self.eps:eps, \
                    self.B_old: B_old, self.r_old: r_old, self.adj: adj_list, self.adj_prev: adj_list_prev, \
                    self.time_cur: time_next}

                    #feed = {self.initial_state_s:initial_state_s, self.initial_state_t:initial_state_t,\
                    #self.input_data: x, self.target_data: y, self.features: features, self.eps:eps, \
                    #self.B_old: B_old, self.r_old: r_old, self.adj: adj_list, self.adj_prev: adj_list_prev, \
                    #self.time_cur: time_next}
                    #P, h_inter_enc, y_t_enc, y_s, y_s_enc, y_current, hidden, a_l, c_l, l_a, l_c, ll_list, kl_loss_zeta_list, kl_loss_z_list, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma = sess.run(
                    #[self.P, self.h_inter_enc, self.y_t, self.y, self.y_s_enc, self.y_current, self.hidden, self.a_l_list, self.c_l_list, self.l_a, self.l_c, self.ll_list, self.kl_loss_zeta_list, self.kl_loss_z_list ,self.enc_zeta_mu, self.enc_z_mu, self.enc_z_sigma, self.enc_zeta_sigma, self.prior_z_mu, self.prior_zeta_mu, self.prior_z_sigma, self.prior_zeta_sigma], feed)

                    kl_loss_z_list, kl_loss_zeta_list, ll_list,  grad, state_list, train_loss, cr, final_state, B_old_1, r_old_1 = sess.run(
                            [self.kl_loss_z_list, self.kl_loss_zeta_list, self.ll_list,  \
                                    self.grads, self.state_list, self.cost, self.train_op, self.final_state, self.B_new, self.r_new], feed)
                    
                    print("Debug loss KL_Z:",kl_loss_z_list)
                    print("Debug loss KL_zeta:", kl_loss_zeta_list)
                    print("Debug loss LL_loss:", ll_list)
                    #np.savetxt("B_"+str(e)+".txt", B_old_1[-1])
                    #np.savetxt("B_hidden_"+str(e)+".txt", B_hidden[-1])
                    #np.savetxt("C_"+str(e)+".txt", C[-1])
                    #np.savetxt("R_"+str(e)+".txt", r_old_1[-1])
                    if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                        checkpoint_path = os.path.join(dirname, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                        print("model saved to {}".format(checkpoint_path))
                    end = time.time()

                    print("{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}" \
                        .format(e * n_batches + b,
                                args.num_epochs * n_batches,
                                e, args.seq_length * train_loss, end - start))
                    start = time.time()
                    B_old = np.reshape(B_old_1[-1], [args.n_c, args.n_c])
                    r_old = np.reshape(r_old_1[-1], [args.n, args.n])

                    #initial_state_s = [initial_state_s[-1]]
                    #initial_state_t = [initial_state_t[-1]]
                    initial_state = final_state[0]
            checkpoint_path = os.path.join(dirname, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=args.num_epochs * n_batches + len(samples))
            np.savetxt(dirname+"/state.txt", initial_state)
            np.savetxt(dirnmae+"/B.txt", B_old)
            np.savetxt(dirname+"/r.txt", r_old)
            #tf.reset_default_graph()



    def get_state(self, args, samples):
        B_old = np.zeros([args.n_c, args.n_c])
        r_old = np.zeros([args.n, args.n])
        #initial_state_t = np.zeros([1, args.h_dim])
        #initial_state_s = np.zeros([1, args.h_dim])
        #initial_state = np.zeros([1, 2 * args.h_dim])
        adj_list = []
        adj_list_prev = []
        features = get_one_hot_features(self.n) 
        initial_state = np.random.rand(1, 2 * args.h_dim)
        adj_old = starting_adj(args, samples)
        eps = np.random.randn(args.n, args.z_dim, 1)
        sess = self.sess
        for b in range(len(samples)):
            x, y = next_batch(args, samples, b)
            # x = np.reshape(x, [])
            time_next = extract_time(args, y)
            if len(adj_list_prev) > 0:
                adj_list_prev[0] = copy.copy(adj_list[-1])
            adj_list = get_adjacency_list(x, adj_old, args.n)

            if len(adj_list_prev) > 0:
                adj_list_prev[1:] = copy.copy(adj_list[:-1])
            else:
                adj_list_prev = [np.zeros((args.n, args.n))]
                adj_list_prev.extend(adj_list[:-1])
            adj_old = adj_list[-1]

            feed = {self.initial_state:initial_state, \
                    self.input_data: x, self.target_data: y, self.features: features, self.eps:eps, \
                    self.B_old: B_old, self.r_old: r_old, self.adj: adj_list, self.adj_prev: adj_list_prev, \
                    self.time_cur: time_next}

            final_state, B_old, r_old, C = sess.run(
                            [self.final_state, self.B_new, self.r_new, self.C], feed)
            
            B_old = np.reshape(B_old[-1], [args.n_c, args.n_c])
            r_old = np.reshape(r_old[-1], [args.n, args.n])
            initial_state = final_state[0]

        return (initial_state, B_old, r_old, adj_old)


    # Experiment 1

    def dynamic_link_prediction(self, args, train_samples, test_samples):
        print("Starting state")
        #state, B_old, r_old, adj_old = self.get_state(args, train_samples[:-1])
        state = np.loadtxt("citation_states/state_c10.txt")
        B_old = np.loadtxt("citation_states/b_c10.txt")
        r_old = np.loadtxt("citation_states/r_c10.txt")
        adj_old = np.loadtxt("citation_states/adj_c10.txt")
        eps = np.loadtxt("eps.txt")
        
        #state = np.loadtxt("github_states/state_g1.txt")
        #B_old = np.loadtxt("github_states/b_g1.txt")
        #r_old = np.loadtxt("github_states/r_g1.txt")
        #adj_old = np.loadtxt("github_states/adj_g1.txt")

        
        #state_s, state_t = state
    
        print("Ending state")
        
        args.sample = True
        initial_state = state
        adj_list = []
        adj_list_prev = []
        
        first_element = train_samples[-1]
        #test_samples = first_element.extend(test_samples)
        test_samples.insert(0, first_element)
        args.sample = True
        hit_a = 0.0
        hit_c = 0.0
        t_a = 0
        t_c = 0
        r_a = 0.0
        r_c = 0.0
    
        ground_truth = []
        l_a_list = []
        l_c_list = []

        features = get_one_hot_features(self.n)
        #adj_old = starting_adj(args, samples)
        #eps = np.random.randn(args.n, args.z_dim, 1)
        sess = self.sess
        initial_state = np.reshape(initial_state, [1, -1])
        for b in range(len(test_samples)):
            x, y = next_batch(args, test_samples, b)
            time_next = extract_time(args, y)
            if len(adj_list_prev) > 0:
                adj_list_prev[0] = copy.copy(adj_list[-1])
            adj_list = get_adjacency_list(x, adj_old, args.n)

            if len(adj_list_prev) > 0:
                adj_list_prev[1:] = copy.copy(adj_list[:-1])
            else:
                adj_list_prev = [np.zeros((args.n, args.n))]
                # print("Debug", len(adj_list_prev))
                adj_list_prev.extend(adj_list[:-1])

            adj_old = adj_list[-1]

            feed = {self.initial_state:initial_state, \
                    self.input_data: x, self.target_data: y, self.features: features, self.eps:eps, \
                    self.B_old: B_old, self.r_old: r_old, self.adj: adj_list, self.adj_prev: adj_list_prev, \
                    self.time_cur: time_next}    
            #print("Shape ", x.shape, r_old.shape)
            state_list, final_state, B_old_1, r_old_1, C, l_a, l_c = sess.run([self.state_list, self.final_state, self.B_new, self.r_new, self.C, self.l_a, self.l_c], feed)
            B_old = np.reshape(B_old_1[-1], [args.n_c, args.n_c])
            r_old = np.reshape(r_old_1[-1], [args.n, args.n])

            #initial_state_s = [initial_state_s[-1]]
            #initial_state_t = [initial_state_t[-1]]
            initial_state = final_state[0]
            ''' 
            hit_, r, true_event = calculate_association(test_samples[b], l_a, self.n)
            hit_a += hit_
            r_a += r
            t_a += true_event
            hit_, r, true_event = calculate_communication(test_samples[b], l_c, self.n)
            hit_c += hit_
            r_c += r
            t_c += true_event
            '''
            ground_truth.extend(y[0])
            l_a_list.extend(l_a)
            l_c_list.extend(l_c)
        get_most_recent_time(args, ground_truth[:100000], self.n, l_a_list[:100000], l_c_list[:100000])
        hit_a, r_a, t_a = get_rank_new(ground_truth, 0)
        hit_c, r_c, t_c = get_rank_new(ground_truth, 1)
        print("Len test sample:", len(test_samples),t_a,t_c)
        print("hit_a:", hit_a)
        print("hit_c:", hit_c)
        print("Total hit", (hit_a + hit_c)/(t_a + t_c))
        print("rank_a:", r_a)
        print("rank_c:", r_c)
        print("Total rank:", (r_a + r_c)/(t_a + t_c))
        return (hit_a + hit_c)


    # Experiment 2

    def event_time_predicion(self, args, train_samples, test_samples):
        #B_old = np.zeros([args.n_c, args.n_c])
        #r_old = np.zeros([args.n, args.n])
        #initial_state_t = np.zeros([1, args.h_dim])
        #initial_state_s = np.zeros([1, args.h_dim])
        #initial_state = np.zeros([1, 2 * args.h_dim])
        #adj_list = []
        #adj_list_prev = []
        #features = get_one_hot_features(self.n)
        #initial_state = np.random.rand(1, 2 * args.h_dim)
        #adj_old = starting_adj(args, samples)
        #eps = np.random.randn(args.n, args.z_dim, 1)
        #sess = self.sess

        state, B_old, r_old, adj_old = self.get_state(args, train_samples[:-1])
        initial_state = state
        adj_list = []
        adj_list_prev = []
        
        first_element = train_samples[-1]
        #test_samples = first_element.extend(test_samples)
        test_samples.insert(0, first_element)
        #args.sample = True
        hit_a = 0
        hit_c = 0
        features = get_one_hot_features(self.n) 
        #adj_old = starting_adj(args, train_samples)
        eps = np.random.randn(args.n, args.z_dim, 1)
        sess = self.sess
        l_a_list = []
        l_c_list = []
        ground_truth = []
        #for b in range(100):
        for b in range(len(test_samples)):
            x, y = next_batch(args, test_samples, b)
            time_next = extract_time(args, y)
            if len(adj_list_prev) > 0:
                adj_list_prev[0] = copy.copy(adj_list[-1])
            adj_list = get_adjacency_list(x, adj_old, args.n)

            if len(adj_list_prev) > 0:
                adj_list_prev[1:] = copy.copy(adj_list[:-1])
            else:
                adj_list_prev = [np.zeros((args.n, args.n))]
                adj_list_prev.extend(adj_list[:-1])

            adj_old = adj_list[-1]

            feed = {self.initial_state:initial_state, \
                    self.input_data: x, self.target_data: y, self.features: features, self.eps:eps, \
                    self.B_old: B_old, self.r_old: r_old, self.adj: adj_list, self.adj_prev: adj_list_prev, \
                    self.time_cur: time_next}    

            state_list, final_state, B_old_1, r_old_1, C, l_a, l_c = sess.run([self.state_list, self.final_state, self.B_new, self.r_new, self.C, self.l_a, self.l_c], feed)
            ground_truth.extend(y[0])
            l_a_list.extend(l_a)
            l_c_list.extend(l_c)
            B_old = np.reshape(B_old_1[-1], [args.n_c, args.n_c])
            r_old = np.reshape(r_old_1[-1], [args.n, args.n])

            initial_state = final_state[0]
        print("GT debug", ground_truth[0])
        test_times = [t for (u, v,  t, m) in ground_truth]
        types = [m for (u, v, t, m) in ground_truth]
        expected_c = get_expected_a(test_times, l_c_list, self.n)
        expected_a = get_expected_a(test_times, l_a_list, self.n)
        
        #shift the events by 1
        predicted = []
        #for i in range(2999):
        for i in range(len(types) -1):
            (u, v, m, t) = ground_truth[i + 1]
            #print("U:", i, u, v)
            if m == 1:
                e_t = expected_c[i][int(u)][int(v)]
            else:
                #e_t = 0 
                #if m == 0:
                e_t = expected_a[i][int(u)][int(v)]
            predicted.append(e_t)
        print("predicted:", predicted)
        error = find_error(test_times[1:], predicted)
        print("error", error)
        return error

    # Experiment 3
    # def feature_space_analysis(state, train_samples, test_samples):
    # TBD
