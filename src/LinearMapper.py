import tensorflow as tf


class LinearMapper(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def init_weights(self, shape, name,bias=False):
        if bias == True:
            return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer())
        else:
            return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))


    def model(self, input, p_keep_input,
              p_keep_hidden):
        # this network is the same as the previous one except with an
        # extra hidden layer + dropout
        input = tf.nn.dropout(input, p_keep_input)

        h = tf.matmul(input, self.w_h) + self.b_h
        h = tf.nn.dropout(h, p_keep_hidden)

        return tf.sigmoid(tf.matmul(h, self.w_o) + self.b_o), h

    def build_mapping_model(self):
        self.input_states_batch = tf.placeholder("float", [self.hparams.linear_steps,None,self.hparams.input_dim])
        #for i in np.arange(self.hparams.linear_steps):
        #    self.input_states_batch_steps[i] = tf.placeholder("float", [None,self.hparams.input_dim])
        self.output_states_batch = tf.placeholder("float", [None, self.hparams.output_dim])
        self.p_keep_input = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.batch_size = tf.placeholder("int32")

        with tf.variable_scope("step_weights"):
            self.step_weights = tf.get_variable(name="w_step",shape=(self.hparams.linear_steps), initializer=tf.ones_initializer())

        #self.step_weights = tf.nn.softmax(self.step_weights)
        #self.input_states_batch_steps = tf.unstack(self.input_states_batch_steps)
        weighted_input_states_batch = [w_step * input_step for input_step, w_step in  zip(tf.unstack(self.input_states_batch),tf.unstack(self.step_weights))]
        self.input_states_batch_combined = tf.reduce_mean(weighted_input_states_batch,axis=0)
        #self.input_states_batch = tf.reduce_mean([ self.input_states_batch_steps[i] * step_weights[i] for i in np.arange(self.hparams.linear_steps)],axis=0)
        tf.logging.info('input shape')
        tf.logging.info(self.input_states_batch.shape)
        with tf.variable_scope("hidden_layers"):
            self.w_h = self.init_weights(name="w_h",shape=[self.hparams.input_dim, self.hparams.hidden_dim])
            self.b_h = self.init_weights(name="b_h",shape=[self.hparams.hidden_dim],bias=True)

            self.variable_summaries(self.w_h, "w_h")
            self.variable_summaries(self.b_h, "b_h")

        with tf.variable_scope("output_layer"):
            self.w_o = self.init_weights(name="w_o",shape=[self.hparams.hidden_dim, self.hparams.output_dim])
            self.b_o = self.init_weights(name="b_o",shape=[self.hparams.output_dim],bias=True)

            self.variable_summaries(self.w_o, "w_o")
            self.variable_summaries(self.b_o, "b_o")

        self.predicted_output, self.h = self.model(self.input_states_batch_combined, self.p_keep_input, self.p_keep_hidden)
        #self.predicted_output = tf.nn.l2_normalize(self.predicted_output, dim=1)
        self.mean_error, self.sd_error = tf.nn.moments(tf.subtract(self.predicted_output, self.output_states_batch), axes=[1,0])

        tf.summary.histogram("predicted_outputs", self.predicted_output)

        #self.cost = tf.reduce_mean(tf.losses.cosine_distance(predictions=self.predicted_output, labels=self.output_states_batch,dim=1)) 
        self.mean_squared_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.output_states_batch, predictions=self.predicted_output))
        self.pred_dists = tf.losses.mean_pairwise_squared_error(labels=self.predicted_output,predictions=self.predicted_output)
        self.target_dists = tf.losses.mean_pairwise_squared_error(labels=self.output_states_batch,predictions=self.output_states_batch)
        self.corelation_loss = 0.0001 * tf.reduce_mean(tf.abs(self.pred_dists - self.target_dists))
        self.hidden_dists = tf.losses.mean_pairwise_squared_error(labels=self.h,predictions=self.h)
        self.descrimination_loss = 0.001 * tf.reduce_mean(tf.abs(self.hidden_dists - self.pred_dists))

        all_vars = [self.w_h, self.w_o, self.step_weights]
        
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars ]) * 0.0001
        #tf.summary.scalar("sigmoid_loss", self.cost)
        tf.summary.scalar("mse", self.mean_squared_loss)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=self.global_step,
            decay_steps=self.hparams.training_size,
            decay_rate=0.95,
            staircase=True)
        tf.summary.scalar("learning_rate", self.learning_rate)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize( self.mean_squared_loss + self.l2_loss, global_step=self.global_step)

        self.summ_op = tf.summary.merge_all()

    def variable_summaries(self, var, name):
        """
        Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        """
        with tf.name_scope(name + '_summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)
