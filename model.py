import tensorflow as tf
from utils import *
from cell_new import MPPCell
import time 
from datetime import datetime
import numpy as np

class MPPModel():
    def __init__(self, args, sample=False):

        self.n = args.n

        def tf_likelihood(y, l_a, l_c):
            with tf.variable_scope('likelihood'):
                u = y[0][0]
                v = y[0][1]
                m = y[0][3]
                t = y[0][2]

                # zero one vector for the events to occur
                # event that did not occur

                #comp_y = tf.subtract(tf.ones([self.n, self.n]), y)
                
                #l_a = tf.reshape(l_a, [self.n, self.n])
                #l_c = tf.reshape(l_c, [self.n, self.n])
                
                index = self.n * (u-1) + v
                
                l_a_occured = tf.cast([1 - m], tf.float32) * tf.gather(l_a, index)
                l_c_occured = tf.cast([m], tf.float32) * tf.gather(l_c, index)

                #l_a_occured = tf.cast([1 - m], tf.float32) * tf.gather_nd(l_a, [u, v])
                #l_c_occured = tf.cast([m], tf.float32) * tf.gather_nd(l_c, [u, v])


                l_a_comp = tf.reduce_sum(l_a) - l_a_occured
                l_c_comp = tf.reduce_sum(l_c) - l_c_occured

                #l_a_occured = tf.log(tf.multiply(y, l_a))
                #l_a_comp = tf.multiply(comp_y, l_a) 
                
                #l_c_occured = tf.log(tf.multiply(y, l_c))
                #l_c_comp = tf.multiply(comp_y, l_c)
                
                #indicator = (1 - m)
                
                association_loss =  tf.cast([1 - m], tf.float32) * tf.subtract(tf.log(tf.maximum(l_a_occured, 1e-09)), tf.add(l_a_comp, l_c_comp))
                communication_loss = tf.cast([m], tf.float32) * tf.subtract(tf.log(tf.maximum(l_c_occured, 1e-09)), tf.add(l_a_comp, l_c_comp)) 
                
                #print "Debug shapes", l_a_occured.get_shape(), l_c_occured.get_shape(), l_a_comp.get_shape(), l_c_comp.get_shape(), association_loss.get_shape(), communication_loss.get_shape()
                #print "Debug c", communication_loss.get_shape()
                #print "Debug a", association_loss.get_shape()
                
		ll = association_loss + communication_loss 
                tf.summary.tensor_summary('ll', ll)
                
                return (association_loss, communication_loss, ll)

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            
            k = tf.fill([self.n], tf.cast(args.z_dim, tf.float32))
            #print "debug mu, sigma", mu_1, sigma_1
	    #print "debug mu, sigma", mu_2, sigma_2
	   
	    with tf.variable_scope("kl_gaussisan"):
                sigma_2_sigma_1 = []
                sigma_mu_sigma = []
                det = []
                
                for i in range(self.n):
                    sigma_2_inv = tf.linalg.inv(sigma_2[i])
                    sigma_2_sigma_1.append(tf.trace(tf.multiply( sigma_2_inv, sigma_1[i])))
                    mu_diff = tf.subtract(mu_1[i], mu_2[i])
                    sigma_mu_sigma.append(tf.matmul(tf.matmul(tf.transpose(mu_diff), sigma_2_inv), mu_diff))
                    det.append(tf.log(tf.maximum(tf.linalg.det(sigma_2), 1e-09)) - tf.log(tf.maximum(tf.linalg.det(sigma_1), 1e-09)))
                
		first_term = tf.stack(sigma_2_sigma_1)
                second_term = tf.stack(sigma_mu_sigma)
                third_term = tf.stack(det)
                #print "Debug size", first_term.get_shape(), second_term.get_shape(), third_term.get_shape()
                k = tf.fill([self.n], tf.cast(args.z_dim, tf.float32))
                return tf.reduce_sum(0.5 *(first_term + second_term + (third_term) - k))


        
        def get_lossfunc(l_c, l_a, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma, y):
            loss = 0.0
            #print "Debug size enc", enc_zeta_mu.get_shape(), y.get_shape()
            #y = y[0]
            y = tf.transpose(y, [1,0,2] )
            
            self.kl_loss_zeta_list = []
	    self.kl_loss_z_list = []
	    self.ll_list = []
	    self.a_l_list = []
	    self.c_l_list = []

	    for i in range(args.seq_length):
                #print "Debug i", i, l_c[i], l_a[i]
                kl_loss_zeta = tf_kl_gaussgauss(enc_zeta_mu[i], enc_zeta_sigma[i], prior_zeta_mu[i], prior_zeta_sigma[i])
                kl_loss_z = tf_kl_gaussgauss(enc_z_mu[i], enc_z_sigma[i], prior_z_mu[i], prior_z_sigma[i])
                self.kl_loss_zeta_list.append(kl_loss_zeta)
		self.kl_loss_z_list.append(kl_loss_z)
		a_l, c_l, likelihood_loss = tf_likelihood(y[i],l_a[i], l_c[i])
                self.ll_list.append(likelihood_loss)
		self.a_l_list.append(a_l)
		self.c_l_list.append(c_l)
		loss += tf.reduce_mean(kl_loss_zeta + kl_loss_z - likelihood_loss)
            return loss

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
        with tf.device('/device:GPU:1'):
         cell = MPPCell(args, self.features, self.B_old, self.r_old, self.eps)
        
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
        
        with tf.variable_scope("RNN"):
            for time_step in range(args.seq_length):
                    #if time_step > 0:
                    #print "time step", time_step
                    tf.get_variable_scope().reuse_variables()
                    (cell_output, state_t, state_s) = self.cell.new_call(self.input_data[:, time_step, :], self.adj[time_step], self.adj_prev[time_step], self.time_cur[:,time_step, :], state)
                    state = (state_t, state_s)
                    #print "debug output", cell_output
                    outputs.append(cell_output)
        #outputs, last_state = self.cell.new_call(self.input_data, list_dubug)
        #outputs, last_state = tf.contrib.rnn.static_rnn(self.cell, inputs, initial_state=list_debug)
        
        outputs_reshape = []
        names = ["y", "y_cuurent", "l_c", "l_a", "enc_zeta_mu", "enc_z_mu", "enc_z_sigma", "enc_zeta_sigma", "prior_z_mu", "prior_zeta_mu", "prior_z_sigma", "prior_zeta_sigma", "B", "r"]
        
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                #x = tf.transpose(x,[1,0,2])
                #x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)

        self.y, self.y_current, l_c, l_a, self.enc_zeta_mu, self.enc_z_mu, self.enc_zeta_sigma, self.enc_z_sigma, self.prior_zeta_mu, self.prior_z_mu, self.prior_zeta_sigma, self.prior_z_sigma, self.B_new, self.r_new = outputs_reshape
        self.final_state_t, self.final_state_s = state
        #self.mu = dec_mu
        #self.sigma = dec_sigma
        #print "Debug size before the lossfunc", enc_zeta_mu
        lossfunc = get_lossfunc(l_c, l_a, self.enc_zeta_mu, self.enc_z_mu, self.enc_z_sigma, self.enc_zeta_sigma, self.prior_z_mu, self.prior_zeta_mu, self.prior_z_sigma, self.prior_zeta_sigma, self.target_data)
        
        self.l_c = l_c
        self.l_a = l_a

        with tf.variable_scope('cost'):
            self.cost = lossfunc 
        
        #print 'cost', self.cost
        #print 'lambda_communication', l_c
        #print 'lambda_association', l_a

        tf.summary.tensor_summary('cost', self.cost)
        tf.summary.tensor_summary('lambda_communication', l_c)
        tf.summary.tensor_summary('lambda_association', l_a)

        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        print_vars("trainable_variables")
        #for t in tvars:
        #    print "trainable vars", t.name
        t1 = time.time()
        grads = tf.gradients(self.cost, tvars)
        t2 = time.time()
        print "After grad:", t2 - t1
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        

    def sample(self):
        #TBD
        return ""

    def train(self, args, samples):

        dirname = args.out_dir
        #print "Time size of graph", len(tf.get_default_graph().get_operations())

        ckpt = tf.train.get_checkpoint_state(dirname)

        n_batches = len(samples)
        #sample_train, sample_test = create_sample(args)
        config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False      
	)
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
        #with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
            #check = tf.add_check_numerics_ops()
            merged = tf.summary.merge_all()
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
        
            initial_state_t = np.zeros([1, args.h_dim])
            initial_state_s = np.zeros([1, args.h_dim])
            features = get_one_hot_features(self.n)
            
            eps = np.random.randn(args.n, args.z_dim, 1)
            B_old = np.zeros([args.n_c, args.n_c])
            r_old = np.zeros([args.n, args.n])

            adj_old = starting_adj(args, samples)
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print "Loaded model"
            start = time.time()
            for e in xrange(args.num_epochs):
                print "Size of graph", len(tf.get_default_graph().get_operations())
		sess.run(tf.assign(self.lr, args.learning_rate * (args.decay_rate ** e)))                
                adj_list = [] 
                adj_list_prev = []
                for b in xrange(len(samples)):
                    x, y = next_batch(args, samples, b)
                    #x = np.reshape(x, [])
                    time_next = extract_time(args, y)
                    if len(adj_list_prev) > 0:
                        adj_list_prev[0] = adj_list[-1]
                    adj_list = get_adjacency_list(x, adj_old, self.n)

                    if len(adj_list_prev) > 0:
                        adj_list_prev[1:] = adj_list[:-1]
                    else:
                        adj_list_prev = [np.zeros((self.n, self.n))].append(adj_list[:-1])
		    
		    #print "Debug adj list_prev:", adj_list_prev
		    #print "Debug adj_list:", adj_list
		    
		    print "Debug adj_list_prev:"
		    for i_index in range(84):
			print adj_list_prev[0][i_index]

                    print "Debug adj_list:"
                    for i_index in range(84):
                        print adj_list[0][i_index]
		    
		    adj_old = adj_list[-1]
             
                    feed = {self.initial_state_s:initial_state_s, self.initial_state_t:initial_state_t,\
                    self.input_data: x, self.target_data: y, self.features: features, self.eps:eps, \
                    self.B_old: B_old, self.r_old: r_old, self.adj: adj_list, self.adj_prev: adj_list_prev, \
                    self.time_cur: time_next}
                    y, y_current, a_l, c_l, l_a, l_c, ll_list, kl_loss_zeta_list, kl_loss_z_list, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma = sess.run(
		    [self.y, self.y_current, self.a_l_list, self.c_l_list, self.l_a, self.l_c, self.ll_list, self.kl_loss_zeta_list, self.kl_loss_z_list ,self.enc_zeta_mu, self.enc_z_mu, self.enc_z_sigma, self.enc_zeta_sigma, self.prior_z_mu, self.prior_zeta_mu, self.prior_z_sigma, self.prior_zeta_sigma], feed)
		    #mu_enc, sigma_enc, prior_mu, prior_sigma = sess.run(
		    '''
		    print "Debug l_a:", l_a
		    print "Debug l_c:", l_c
		    print "Debug association_loss:", a_l
		    print "Debug communication_loss:", c_l
		    print "Debug y:", y
		    print "Debug y_current:", y_current
		    print "Debug ll_list:", ll_list
		    print "Debug kl_zeta:", kl_loss_zeta_list
		    print "Debug kl_z:", kl_loss_z_list
		    print "Debug enc_zeta_mu:", enc_zeta_mu
		    print "Debug enc_z_mu:", enc_z_mu 
		    print "Debug enc_z_sigma:", enc_z_sigma
		    print "Debug enc_zeta_sigma:", enc_zeta_sigma
		    print "Debug prior_z_mu:", prior_z_mu
		    print "Debug prior_zeta_mu:", prior_zeta_mu 
		    print "Debug prior_z_sigma:", prior_z_sigma
		    print "Debug prior_zeta_sigma:", prior_zeta_sigma
		    '''

		    train_loss, cr, summary, initial_state_s, initial_state_t, B_old, r_old = sess.run(
                            [self.cost, self.train_op, merged, self.final_state_s, self.final_state_t, self.B_new, self.r_new], feed)
                    summary_writer.add_summary(summary, e * n_batches + b)
                    if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                        checkpoint_path = os.path.join(dirname, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                        print "model saved to {}".format(checkpoint_path)
                    end = time.time()
                    print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}" \
                        .format(e * n_batches + b,
                                args.num_epochs * n_batches,
                                e, args.seq_length * train_loss, end - start)
                    start = time.time()
		    B_old = np.reshape(B_old, [args.n_c, args.n_c])
		    r_old = np.reshape(r_old, [args.n, args.n])

    def predict_association(self, test_sample, adj):
        for b in xrange(len(samples)):
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


