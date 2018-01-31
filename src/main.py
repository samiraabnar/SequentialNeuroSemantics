import tensorflow as tf
import numpy as np
import random
from collections import namedtuple
import numpy as np


import math
import os

from HarryPotterDataProcessing import *
from train import *




FLAGS = tf.app.flags.FLAGS

# ========Where to save outputs===========
tf.app.flags.DEFINE_string('log_root', '../log_root', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('data_path', '../data', 'Directory where the data '
                                                   'is going to be saved.')
tf.app.flags.DEFINE_string('mapper', 'decoder', 'intended/forward/decoder')
tf.app.flags.DEFINE_string('exp_name', 'row_norm_MSE_lossNoAttention_smallLR_lessreg_relu_concat_cosine_dist', 'Name for experiment. Logs will '
                                                          'be saved in a directory with this'
                                                          ' name, under log_root.')
tf.app.flags.DEFINE_string('model', 'char_word', 'must be one of '
                                               'char_word/word/contextual_0/contextual_1/contextual_01/char/glove')
tf.app.flags.DEFINE_string('direction', 'word2brain', 'must be one of '
                                               'brain2word/word2brain')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of '
                                               'train/test/save_vectors')
tf.app.flags.DEFINE_string('timeshift', '0', 'must be a positive or negetive integer')

# ==========Hyper Params=========
tf.app.flags.DEFINE_integer('batch_size', 50, 'minibatch size')
tf.app.flags.DEFINE_integer('hidden_dim', 512, 'dimension of hidden states')
tf.app.flags.DEFINE_integer('input_dim', 784, 'size of the input')
tf.app.flags.DEFINE_integer('output_dim', 784, 'size of the output')

# ===== Training Setup=======
tf.app.flags.DEFINE_integer('number_of_epochs', 20, 'number_of_epochs')
tf.app.flags.DEFINE_integer('training_size', 20, 'training_size')





def prepare(FLAGS):
    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want

    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.direction, FLAGS.model, FLAGS.mapper, FLAGS.exp_name, )
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)

    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = load_data(FLAGS)

    hps = compile_params(train_embeddings, train_normalized_brain_scans, train_size)

    return hps, test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_words

def compile_params(train_embeddings, train_normalized_brain_scans, train_size):
    if FLAGS.direction == "word2brain":
        FLAGS.input_dim = train_embeddings.shape[1]
        FLAGS.output_dim = train_normalized_brain_scans.shape[1]
    elif FLAGS.direction == "brain2word":
        FLAGS.output_dim = train_embeddings.shape[1]
        FLAGS.input_dim = train_normalized_brain_scans.shape[1]
        print("input dim:",FLAGS.input_dim)

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

    columns_min = np.min(train_normalized_brain_scans,axis=0)
    columns_max = np.max(train_normalized_brain_scans,axis=0)
    print("min max",columns_max - columns_min)
    train_normalized_brain_scans = (train_normalized_brain_scans - columns_min) / (columns_max - columns_min + 0.000001)

    print("max:",np.max(train_normalized_brain_scans))
    print("min:",np.min(train_normalized_brain_scans))

    columns_min = np.min(test_normalized_brain_scans,axis=0)
    columns_max = np.max(test_normalized_brain_scans,axis=0)
    print("min max",columns_max - columns_min)
    test_normalized_brain_scans = (test_normalized_brain_scans - columns_min) / (columns_max - columns_min + 0.000001)



    if FLAGS.mapper == "intended":
        from VanillaIntendedMapper import VanillaIntendedMapper as StateMapper
    elif FLAGS.mapper == "decoder":
        from DecodedMapper import DecodedMapper as StateMapper
    else:
        from StateMapper import StateMapper as StateMapper


    with tf.Graph().as_default():
        # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
        train_dir = os.path.join(FLAGS.log_root, "train")
        if not os.path.exists(train_dir): os.makedirs(train_dir)


        mapper = StateMapper(hps)
        mapper.build_mapping_model()

        saver = tf.train.Saver(max_to_keep=3)
        sv = tf.train.Supervisor(logdir=train_dir,
                                 is_chief=True,
                                 saver=saver,
                                 summary_op=None,
                                 save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                                 save_model_secs=60  # checkpoint every 60 secs
                                 )

        # Get a TensorFlow session managed by the supervisor.
        with sv.managed_session() as sess:
            if FLAGS.direction == "word2brain":
                if FLAGS.mode == "train":
                    train(mapper, sess, sv, train_embeddings, train_normalized_brain_scans,
                      test_embeddings, test_normalized_brain_scans,
                      test_words=test_words,train_words=train_words,FLAGS=FLAGS)
                elif FLAGS.mode == "save_vectors":
                    print("saving vectors...")
                    save_pred_and_target_labesl(mapper, sess,test_embeddings,test_normalized_brain_scans,test_words,FLAGS)


            elif FLAGS.direction == "brain2word":
                if FLAGS.mode =="train":
                    print("brain2word training...")
                    train(mapper, sess, sv, train_normalized_brain_scans, train_embeddings,
                        test_normalized_brain_scans, test_embeddings,
                        test_words=test_words, train_words=train_words, FLAGS=FLAGS)
                elif FLAGS.mode == "save_vectors":
                  print("saving vectors...")
                  save_pred_and_target_labesl(mapper, sess, test_normalized_brain_scans, test_embeddings,test_words,FLAGS)



if __name__ == '__main__':
    tf.app.run()
