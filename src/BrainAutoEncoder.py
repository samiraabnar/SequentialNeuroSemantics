import tensorflow as tf


import os
from train_autoencoder import train
from HarryPotterDataProcessing import *
from collections import namedtuple



class BrainAutoEncoder(object):
    def __init__(self,hparams):
		    self.hparams = hparams

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

    def init_weights(self, shape, name,bias=False):
        if bias == True:
            return tf.get_variable(name=name, shape=shape, initializer=tf.zeros_initializer())
        else:
            return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))


    def encode(self,input):
        return tf.matmul(input,self.w_in) + self.b_in


    def decode(self,hidden_state):
        return tf.sigmoid(tf.matmul(hidden_state,self.w_out) + self.b_out)

    def build_graph(self):
        self.input_states_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.input_dim])
        self.output_states_batch = tf.placeholder(dtype=tf.float32, shape=[None, self.hparams.input_dim])

        self.p_keep_input = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.batch_size = tf.placeholder("int32")

        with tf.variable_scope("encoder_layer"):
            self.w_in = self.init_weights(name="w_in",shape=[self.hparams.input_dim, self.hparams.hidden_dim])
            self.b_in = self.init_weights(name="b_in",shape=[self.hparams.hidden_dim],bias=True)
            self.variable_summaries(self.w_in, "w_in")
            self.variable_summaries(self.b_in, "b_in")


        with tf.variable_scope("decoder_layaer"):
            self.w_out = self.init_weights(name="w_out",shape=[self.hparams.hidden_dim, self.hparams.output_dim])
            self.b_out = self.init_weights(name="b_out",shape=[self.hparams.output_dim],bias=True)
            self.variable_summaries(self.w_out, "w_out")
            self.variable_summaries(self.b_out, "b_out")




        self.encoded_input = self.encode(self.input_states_batch)
        self.decoded_output = self.decode(self.encoded_input)
        self.predicted_output = self.decoded_output
        self.mean_error, self.sd_error = tf.nn.moments(tf.subtract(self.predicted_output, self.input_states_batch), axes=[1,0])

        self.mean_squared_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.input_states_batch, predictions=self.decoded_output))
        all_vars = [self.w_in, self.w_out]
        self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in all_vars ]) * 0.001
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
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize( self.mean_squared_loss, global_step=self.global_step)
        self.summ_op = tf.summary.merge_all()



tf.set_random_seed(1234)

FLAGS = tf.app.flags.FLAGS

# ========Where to save outputs===========
tf.app.flags.DEFINE_string('log_root', '../log_root/BrainAutoEncoder/', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('data_path', '../data', 'Directory where the data '
                                                   'is going to be saved.')
tf.app.flags.DEFINE_string('exp_name', 'simple_drop_connect', 'Name for experiment. Logs will '
                                                          'be saved in a directory with this'
                                                          ' name, under log_root.')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of '
                                               'train/test/save_vectors')
tf.app.flags.DEFINE_string('select', '0', 'must be a positive integer')

# ==========Hyper Params=========
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('hidden_dim', 512, 'dimension of hidden states')
tf.app.flags.DEFINE_integer('input_dim', 784, 'size of the input')
tf.app.flags.DEFINE_integer('output_dim', 784, 'size of the output')
tf.app.flags.DEFINE_float('p_keep_input',0.9,'positive float')
tf.app.flags.DEFINE_float('p_keep_hidden',0.6,'positive float')
# ===== Training Setup=======
tf.app.flags.DEFINE_integer('number_of_epochs', 20, 'number_of_epochs')
tf.app.flags.DEFINE_integer('training_size', 20, 'training_size')
tf.app.flags.DEFINE_string('model','autoencoder', 'autoencoder')



def prepare(FLAGS):
    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)

    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = load_data(FLAGS)

    hps = compile_params(train_embeddings, train_normalized_brain_scans, train_size)

    return hps, test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_words

def compile_params(train_embeddings, train_normalized_brain_scans, train_size):
    FLAGS.input_dim = train_normalized_brain_scans.shape[1]
    FLAGS.output_dim = train_normalized_brain_scans.shape[1]


    FLAGS.training_size = train_size
    hparam_list = ['batch_size', 'hidden_dim', 'input_dim', 'output_dim', 'number_of_epochs', 'training_size', 'mode']

    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
    return hps


def main(unused_argv):
    hps, test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_words = prepare(FLAGS)


    with tf.Graph().as_default():
        # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
        train_dir = os.path.join(FLAGS.log_root, "train")
        best_dir = os.path.join(FLAGS.log_root, "best")
        if not os.path.exists(train_dir): os.makedirs(train_dir)
        if not os.path.exists(best_dir): os.makedirs(best_dir)


        mapper = BrainAutoEncoder(hps)
        mapper.build_graph()

        saver = tf.train.Saver(max_to_keep=3)
        best_saver = tf.train.Saver(max_to_keep=1)
        sv = tf.train.Supervisor(logdir=train_dir,
                                 is_chief=True,
                                 saver=saver,
                                 summary_op=None,
                                 save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                                 save_model_secs=60  # checkpoint every 60 secs
                                 )

        # Get a TensorFlow session managed by the supervisor.
        with sv.managed_session() as sess:
            train(mapper, sess, sv, train_normalized_brain_scans,train_normalized_brain_scans,train_words, train_words,FLAGS=FLAGS,best_saver=best_saver,best_dir=best_dir)



if __name__ == '__main__':
    tf.app.run()
