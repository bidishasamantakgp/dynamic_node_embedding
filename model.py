import tensorflow as tf
from utils import *
from cell_new import MPPCell

class MPPModel():
    def __init__(self, args, n, sample=False):

        self.n = n

        def tf_likelihood(y, l_a, l_c):
            with tf.variable_scope('likelihood'):
                u = y[0][0]
                v = y[0][1]
                m = y[0][2]
                t = y[0][3]

                #del_t = t - prev
                ll = tf.cast([1 - m], tf.float32) * (tf.log(tf.gather_nd(l_a, (u, v)) - tf.gather_nd(l_a, (u, v)))) + tf.cast((1-m), tf.float32) * (tf.log(tf.gather_nd(l_c, (u, v)) - tf.gather_nd(l_c, (u, v))))
                return ll

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            
            k = tf.fill([self.n], tf.cast(args.z_dim, tf.float32))
            with tf.variable_scope("kl_gaussisan"):
                sigma_1_sigma_2 = []
                sigma_mu_sigma = []
                det = []
                
                for i in range(self.n):
                    sigma_1_inv = tf.linalg.inv(sigma_2[i])
                    sigma_1_sigma_2.append(tf.trace(tf.multiply( sigma_1_inv, sigma_1[i])))
                    mu_diff = tf.subtract(mu_1[i], mu_2[i])
                    sigma_mu_sigma.append(tf.matmul(tf.matmul(tf.transpose(mu_diff), sigma_1_inv), mu_diff))
                    det.append(tf.log(tf.maximum(tf.linalg.det(sigma_2), 1e-09) - tf.log(tf.maximum(tf.linalg.det(sigma_1), 1e-09))))
                first_term = tf.stack(sigma_1_sigma_2)
                second_term = tf.stack(sigma_mu_sigma)
                third_term = tf.stack(det)
                print "Debug size", first_term.get_shape(), second_term.get_shape(), third_term.get_shape()
                k = tf.fill([self.n], tf.cast(args.z_dim, tf.float32))
                return tf.reduce_sum(0.5 *(first_term + second_term + (third_term) - k))


        
        def get_lossfunc(l_c, l_a, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma, y):
            loss = 0.0
            print "Debug size enc", enc_zeta_mu.get_shape(), y.get_shape()
            #y = y[0]
            for i in range(args.seq_length):
                print "Debug i", i
                kl_loss_zeta = tf_kl_gaussgauss(enc_zeta_mu[i], enc_zeta_sigma[i], prior_zeta_mu[i], prior_zeta_sigma[i])
                kl_loss_z = tf_kl_gaussgauss(enc_z_mu[i], enc_z_sigma[i], prior_z_mu[i], prior_z_sigma[i])
                likelihood_loss = tf_likelihood(y[:,i, :], l_a[i], l_c[i])
                loss += tf.reduce_mean(kl_loss_zeta - likelihood_loss)
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
        self.features = tf.placeholder(dtype=tf.float32, shape=[args.n, args.d_dim], name='features')
        self.eps = tf.placeholder(dtype=tf.float32, shape=[args.n, args.z_dim, 1], name='eps')
        self.adj = tf.placeholder(dtype=tf.float32, shape=[args.n, args.n], name='adj')
        self.adj_prev = tf.placeholder(dtype=tf.float32, shape=[args.n, args.n], name='prev')
        self.B_old = tf.placeholder(dtype=tf.float32, shape=[args.n_c, args.n_c], name='B')
        self.r_old = tf.placeholder(dtype=tf.float32, shape=[args.n, args.n], name='R')
        self.time_old = tf.placeholder(dtype=tf.int32, shape=[args.batch_size, args.seq_length, 1], name='T')
        
        #Trial
        self.initial_state_t = tf.placeholder(dtype = tf.float32, shape = [1, args.h_dim], name = "s_t")
        self.initial_state_s = tf.placeholder(dtype = tf.float32, shape = [1, args.h_dim], name = "s_c")
        self.event_indicator = tf.placeholder(dtype = tf.int32, shape = [args.n], name = "indicator")
        cell = MPPCell(args, self.adj, self.adj_prev, self.features, self.B_old, self.r_old, self.eps)
        
        self.cell = cell
        #debug_state_size = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        #print "Debug state size", len(debug_state_size), debug_state_size
        list_debug = (self.initial_state_t, self.initial_state_s)
        print "Debug input", tf.stack(list_debug), list_debug
        #[self.initial_state_t, self.initial_state_s] = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
            inputs = tf.reshape(inputs, [-1, 4]) # (n_steps*batch_size, n_input)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
        inputs = tf.split(axis=0, num_or_size_splits=args.seq_length, value=inputs) # n_steps * (batch_size, n_hidden)
        print "Debug input shpae", self.input_data
        #Get vrnn cell output
        #outputs, last_state = tf.nn.dynamic_rnn(cell=self.cell, dtype=tf.float32, inputs=self.input_data, initial_state=list_debug)
        
        state = (self.initial_state_t, self.initial_state_s)
        outputs = []
        
        with tf.variable_scope("RNN"):
            for time_step in range(args.seq_length):
                    #if time_step > 0:
                    print "time step", time_step
                    tf.get_variable_scope().reuse_variables()
                    (cell_output, state_h, state_t) = self.cell.new_call(self.input_data[:, time_step, :], self.time_old[:,time_step, :], state)
                    state = (state_h, state_t)
                    print "debug output", cell_output
                    outputs.append(cell_output)
        #outputs, last_state = self.cell.new_call(self.input_data, list_dubug)
        #outputs, last_state = tf.contrib.rnn.static_rnn(self.cell, inputs, initial_state=list_debug)
        
        outputs_reshape = []
        names = ["l_c", "l_a", "enc_zeta_mu", "enc_z_mu", "enc_z_sigma", "enc_zeta_sigma", "prior_z_mu", "prior_zeta_mu", "prior_z_sigma", "prior_zeta_sigma"]
        
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                #x = tf.transpose(x,[1,0,2])
                #x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)

        l_c, l_a, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma = outputs_reshape
        self.final_state_c,self.final_state_h = state
        #self.mu = dec_mu
        #self.sigma = dec_sigma
        print "Debug size before the lossfunc", enc_zeta_mu
        lossfunc = get_lossfunc(l_c, l_a, enc_zeta_mu, enc_z_mu, enc_z_sigma, enc_zeta_sigma, prior_z_mu, prior_zeta_mu, prior_z_sigma, prior_zeta_sigma, self.target_data)
        
        self.l_c = l_c
        self.l_a = l_a

        with tf.variable_scope('cost'):
            self.cost = lossfunc 
        
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('lambda_communication', l_c)
        tf.summary.scalar('lambda_association', l_a)


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for t in tvars:
            print "trainable vars", t.name
        grads = tf.gradients(self.cost, tvars)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self):
        #TBD
        return ""

    def train(self, args):
        dirname = args.dirname

        ckpt = tf.train.get_checkpoint_state(dirname)
        n_batches = 100
        samples = create_sample(args)

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter('logs/' + datetime.now().isoformat().replace(':', '-'), sess.graph)
            check = tf.add_check_numerics_ops()
            merged = tf.summary.merge_all()
            tf.global_variables_initializer().run()
            saver = tf.train.Saver(tf.global_variables())
            if ckpt:
                saver.restore(sess, ckpt.model_checkpoint_path)
                print "Loaded model"
            start = time.time()
            for e in xrange(args.num_epochs):
                sess.run(tf.assign(model.lr, args.learning_rate * (args.decay_rate ** e)))
                state = model.initial_state_c, model.initial_state_h
                for b in xrange(len(samples)):
                    x, y = next_batch(args, samples, b)
                    feed = {self.input_data: x, self.target_data: y, self.adj: adj}
                    train_loss, _, cr, summary, sigma, mu, input, target= sess.run(
                            [self.cost, self.train_op, check, merged, self.sigma, self.mu, self.flat_input, self.target],
                                                                 feed)
                    summary_writer.add_summary(summary, e * n_batches + b)
                    if (e * n_batches + b) % args.save_every == 0 and ((e * n_batches + b) > 0):
                        checkpoint_path = os.path.join(dirname, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=e * n_batches + b)
                        print "model saved to {}".format(checkpoint_path)
                    end = time.time()
                    print "{}/{} (epoch {}), train_loss = {:.6f}, time/batch = {:.1f}, std = {:.3f}" \
                        .format(e * n_batches + b,
                                args.num_epochs * n_batches,
                                e, args.chunk_samples * train_loss, end - start, sigma.mean(axis=0).mean(axis=0))
            start = time.time()
            


