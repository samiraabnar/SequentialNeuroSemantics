import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt


class StateMapper(object):
    def __init__(self,hparams):
        self.hparams = hparams

    def init_weights(self,shape):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

    def model(self,X,p_keep_input,
              p_keep_hidden):  # this network is the same as the previous one except with an extra hidden layer + dropout
        X = tf.nn.dropout(X, p_keep_input)
        #h = tf.tanh(tf.matmul(X, self.w_h) + self.b_h)

        #h = tf.nn.dropout(h, p_keep_hidden)


        return tf.nn.sigmoid(tf.matmul(X, self.w_o) + self.b_o)


    def build_mapping_model(self):
        self.input_states_batch = tf.placeholder("float", [None, self.hparams.input_dim])
        self.output_states_batch = tf.placeholder("float", [None, self.hparams.output_dim])
        self.p_keep_input = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.batch_size = tf.placeholder("int32")


        #with tf.variable_scope("hidden_layers"):
            #self.w_h = self.init_weights([self.hparams.input_dim, self.hparams.hidden_dim])
            #self.b_h = self.init_weights([self.hparams.hidden_dim])
            #self.w_h2 = self.init_weights([self.hparams.hidden_dim, self.hparams.hidden_dim])
            #self.b_h2 = self.init_weights([self.hparams.hidden_dim])

            #self.variable_summaries(self.w_h)
            #self.variable_summaries(self.b_h)
            #self.variable_summaries(self.w_h2)
            #self.variable_summaries(self.w_h2)


        with tf.variable_scope("output_layer"):
            self.w_o = self.init_weights([self.hparams.input_dim, self.hparams.output_dim])
            self.b_o = self.init_weights([self.hparams.output_dim])

            self.variable_summaries(self.w_o)
            self.variable_summaries(self.b_o)



        self.predicted_output = self.model(self.input_states_batch,self.p_keep_input, self.p_keep_hidden)

        tf.summary.histogram("predicted outputs",self.predicted_output)

        self.mean_squared_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.output_states_batch,predictions=self.predicted_output))
        self.l2_loss = tf.nn.l2_loss(self.w_o) * 0.001

        self.mean_error, self.sd_error = tf.nn.moments(tf.subtract(self.predicted_output, self.output_states_batch), axes=[1,0])

        tf.summary.scalar("mse", self.mean_squared_loss)

        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=0.001,
            global_step=self.global_step,
            decay_steps=self.hparams.training_size,
            decay_rate=0.95,
            staircase=True)
        tf.summary.scalar("learning_rate",self.learning_rate)
        self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.mean_squared_loss + self.l2_loss, global_step=self.global_step)

        self.summ_op = tf.summary.merge_all()

    def variable_summaries(self,var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

