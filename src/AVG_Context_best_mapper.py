import tensorflow as tf


class VanillaIntendedMapper(object):
    def __init__(self, hparams):
        self.hparams = hparams

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    def pairwise_dist(self, a):
        r = tf.reduce_sum(a * a, 1)
        # turn r into column vector
        r = tf.reshape(r, [-1, 1])
        d = r - 2 * tf.matmul(a, tf.transpose(a)) + tf.transpose(r)

        return d


    def pairwise_l2_norm2(self, x, y, scope=None):
        size_x = tf.shape(x)[0]
        size_y = tf.shape(y)[0]
        xx = tf.expand_dims(x, -1)
        xx = tf.tile(xx, tf.stack([1, 1, size_y]))

        yy = tf.expand_dims(y, -1)
        yy = tf.tile(yy, tf.stack([1, 1, size_x]))
        yy = tf.transpose(yy, perm=[2, 1, 0])

        diff = tf.subtract(xx, yy)
        square_diff = tf.square(diff)

        square_dist = tf.reduce_sum(square_diff, 1)

        return square_dist

    def model(self, input, p_keep_input,
              p_keep_hidden):
        # this network is the same as the previous one except with an
        # extra hidden layer + dropout
        input = tf.nn.dropout(input, p_keep_input)

        # input_attention = tf.nn.relu(tf.matmul(input, self.w_I) + self.b_I)
        # attended_input = tf.multiply(input_attention, input)
        # tf.summary.image("attended_input",attended_input)
        # tf.summary.image("w_h",self.w_h)

        """input = tf.layers.batch_normalization(input,
        axis=1,
        center=True,
        scale=False,
        training=(self.hparams.mode == tf.estimator.ModeKeys.TRAIN))
        """

        h = tf.nn.relu(tf.nn.relu(tf.matmul(input, self.w_h) + self.b_h))

        h = tf.nn.dropout(h, p_keep_hidden)
        #h2 = tf.nn.relu(tf.matmul(h, self.w_h2) + self.b_h2)

        #h2 = tf.nn.dropout(h2, p_keep_hidden)

        

        return tf.sigmoid(tf.matmul(h, self.w_o) + self.b_o)

    def build_mapping_model(self):
        self.input_states_batch = tf.placeholder("float", [None, self.hparams.input_dim])
        self.output_states_batch = tf.placeholder("float", [None, self.hparams.output_dim])
        self.p_keep_input = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.batch_size = tf.placeholder("int32")

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

        with tf.variable_scope("hidden_layers"):
            self.w_I = self.init_weights([self.hparams.input_dim, self.hparams.input_dim])
            self.b_I = self.init_weights([self.hparams.input_dim])

            self.w_h = self.init_weights([self.hparams.input_dim, self.hparams.hidden_dim])
            self.b_h = self.init_weights([self.hparams.hidden_dim])
            #self.w_h2 = self.init_weights([self.hparams.hidden_dim, self.hparams.hidden_dim])
            #self.b_h2 = self.init_weights([self.hparams.hidden_dim])

            self.variable_summaries(self.w_I, "w_I")
            self.variable_summaries(self.b_I, "b_I")
            self.variable_summaries(self.w_h, "w_h")
            self.variable_summaries(self.b_h, "b_h")
            #self.variable_summaries(self.w_h2, "w_h2")
            #self.variable_summaries(self.b_h2, "b_h2")

        with tf.variable_scope("output_layer"):
            self.w_o = self.init_weights([self.hparams.hidden_dim, self.hparams.output_dim])
            self.b_o = self.init_weights([self.hparams.output_dim])

            self.variable_summaries(self.w_o, "w_o")
            self.variable_summaries(self.b_o, "b_o")

        self.predicted_output = self.model(self.input_states_batch, self.p_keep_input, self.p_keep_hidden)
        #self.predicted_output = tf.nn.l2_normalize(self.predicted_output, dim=1)
        tf.logging.info("predicted output shape: ")
        tf.logging.info(self.predicted_output.shape)
        self.mean_error, self.sd_error = tf.nn.moments(tf.subtract(self.predicted_output, self.output_states_batch), axes=[1,0])

        tf.summary.histogram("predicted_outputs", self.predicted_output)

        #self.cost = tf.reduce_mean(tf.losses.cosine_distance(predictions=self.predicted_output, labels=self.output_states_batch,dim=1)) 
        self.mean_squared_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.output_states_batch, predictions=self.predicted_output))
        self.pred_dists = tf.losses.mean_pairwise_squared_error(labels=self.predicted_output,predictions=self.predicted_output)
        self.target_dists = tf.losses.mean_pairwise_squared_error(labels=self.output_states_batch,predictions=self.output_states_batch)
        self.descrimination_loss = 0.001 * tf.reduce_mean(tf.abs(self.pred_dists - self.target_dists))
        all_vars = [self.w_h, self.w_o]
        
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
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize( self.mean_squared_loss + self.l2_loss + self.descrimination_loss, global_step=self.global_step)

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
