


class MPPModel():
    def __init__(self, args, sample=False):

        self.n = n


        def tf_normal(y, mu, s):
            with tf.variable_scope('normal'):
                ss = tf.maximum(1e-10,tf.square(s))
                norm = tf.subtract(y[:,:args.chunk_samples], mu)
                z = tf.div(tf.square(norm), ss)
                denom_log = tf.log(2*np.pi*ss, name='denom_log')
                result = tf.reduce_sum(z+denom_log, 1)/2
            return result

        def tf_kl_gaussgauss(mu_1, sigma_1, mu_2, sigma_2):
            with tf.variable_scope("kl_gaussgauss"):
                return tf.reduce_sum(0.5 * (
                    2 * tf.log(tf.maximum(1e-9,sigma_2),name='log_sigma_2') 
                  - 2 * tf.log(tf.maximum(1e-9,sigma_1),name='log_sigma_1')
                  + (tf.square(sigma_1) + tf.square(mu_1 - mu_2)) / tf.maximum(1e-9,(tf.square(sigma_2))) - 1
                ), 1)

        
        def get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma, y):
            kl_loss = tf_kl_gaussgauss(enc_mu, enc_sigma, prior_mu, prior_sigma)
            likelihood_loss = tf_normal(y, dec_mu, dec_sigma, dec_rho)
            return tf.reduce_mean(kl_loss + likelihood_loss)

        self.args = args
        if sample:
            args.batch_size = 1
            args.seq_length = 1
        #TBD
        cell = MPPCell(args.chunk_samples, args.rnn_size, args.latent_size)
        self.cell = cell

        self.input_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2 * args.chunk_samples], name='input_data')
        self.target_data = tf.placeholder(dtype=tf.float32, shape=[args.batch_size, args.seq_length, 2 * args.chunk_samples],name = 'target_data')
        self.initial_state_c, self.initial_state_h = cell.zero_state(batch_size=args.batch_size, dtype=tf.float32)


        # input shape: (batch_size, n_steps, n_input)
        with tf.variable_scope("inputs"):
            inputs = tf.transpose(self.input_data, [1, 0, 2])  # permute n_steps and batch_size
            inputs = tf.reshape(inputs, [-1, 2*args.chunk_samples]) # (n_steps*batch_size, n_input)

            # Split data because rnn cell needs a list of inputs for the RNN inner loop
            inputs = tf.split(axis=0, num_or_size_splits=args.seq_length, value=inputs) # n_steps * (batch_size, n_hidden)     
        
        flat_target_data = tf.reshape(self.target_data,[-1, 2*args.chunk_samples])

        self.target = flat_target_data
        self.flat_input = tf.reshape(tf.transpose(tf.stack(inputs),[1,0,2]),[args.batch_size*args.seq_length, -1])
        self.input = tf.stack(inputs)
        
        # Get vrnn cell output
        outputs, last_state = tf.contrib.rnn.static_rnn(cell, inputs, initial_state=(self.initial_state_c,self.initial_state_h))
        
        #outputs = map(tf.pack,zip(*outputs))
        outputs_reshape = []
        names = ["enc_mu", "enc_sigma", "dec_mu", "dec_sigma", "prior_mu", "prior_sigma", "lambda_association", "lambda_communication"]
        for n,name in enumerate(names):
            with tf.variable_scope(name):
                x = tf.stack([o[n] for o in outputs])
                x = tf.transpose(x,[1,0,2])
                x = tf.reshape(x,[args.batch_size*args.seq_length, -1])
                outputs_reshape.append(x)

        enc_mu, enc_sigma, dec_mu, dec_sigma, prior_mu, prior_sigma, lambda_association, lambda_communication = outputs_reshape
        
        self.final_state_c,self.final_state_h = last_state
        self.mu = dec_mu
        self.sigma = dec_sigma
        self.rho = dec_rho

        lossfunc = get_lossfunc(enc_mu, enc_sigma, dec_mu, dec_sigma, dec_rho, prior_mu, prior_sigma, flat_target_data)
        self.sigma = dec_sigma
        self.mu = dec_mu
        with tf.variable_scope('cost'):
            self.cost = lossfunc 
        tf.summary.scalar('cost', self.cost)
        tf.summary.scalar('mu', tf.reduce_mean(self.mu))
        tf.summary.scalar('sigma', tf.reduce_mean(self.sigma))


        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        for t in tvars:
            print t.name
        grads = tf.gradients(self.cost, tvars)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self):
        #TBD


    def train(self, args):
        dirname = args.dirname
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        with open(os.path.join(dirname, 'config.pkl'), 'w') as f:
            cPickle.dump(args, f)

        ckpt = tf.train.get_checkpoint_state(dirname)
        n_batches = 100
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
                for b in xrange(n_batches):
                    x, y = next_batch(args)
                    feed = {model.input_data: x, model.target_data: y}
                    train_loss, _, cr, summary, sigma, mu, input, target= sess.run(
                            [model.cost, model.train_op, check, merged, model.sigma, model.mu, model.flat_input, model.target],
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
            


