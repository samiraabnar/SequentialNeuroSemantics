import tensorflow as tf
import numpy as np
import random
from collections import namedtuple
import numpy as np
import math
import os
from HarryPotterDataProcessing import *
from eval import *
tf.set_random_seed(1234)
FLAGS = tf.app.flags.FLAGS
# ========Where to save outputs===========
tf.app.flags.DEFINE_string('log_root', '../log_root', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('fold_id', '1', '1/2/3/4')
tf.app.flags.DEFINE_string('data_path', '../data', 'Directory where the data '
                                                   'is going to be saved.')
tf.app.flags.DEFINE_string('mapper', 'decoder', 'intended/forward/decoder')
tf.app.flags.DEFINE_string('exp_name', 'simple_drop_connect', 'Name for experiment. Logs will '
                                                              'be saved in a directory with this'
                                                              ' name, under log_root.')
tf.app.flags.DEFINE_string('model', 'char_word', 'must be one of '
                                                 'char_word/word/contextual_0/contextual_1/contextual_01/char/glove/contextual_01_avg')
tf.app.flags.DEFINE_string('direction', 'word2brain', 'must be one of '
                                                      'brain2word/word2brain')
tf.app.flags.DEFINE_string('mode', 'train', 'must be one of '
                                            'train/test/save_vectors/eval_voxels')
tf.app.flags.DEFINE_string('fMRI_preprocess_mode', 'detrend_filter_std', 'must be one of '
                                            'none/detrend/detrend_filter/detrend_filter_std/mean')
tf.app.flags.DEFINE_string('timeshift', '0', 'must be a positive or negetive integer')
tf.app.flags.DEFINE_string('select', '0', 'must be a positive integer')
tf.app.flags.DEFINE_string('features', 'selected', 'dim_reducted/selected')
tf.app.flags.DEFINE_string('subject_id', '1', '1-8')

tf.app.flags.DEFINE_integer('ith_word', '-1', 'which word to look at (-1 means all!')
tf.app.flags.DEFINE_integer('ith_step', '0', 'which word to look at (0 means current step!')

# ==========Hyper Params=========
tf.app.flags.DEFINE_integer('batch_size', 32, 'minibatch size')
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of hidden states')
tf.app.flags.DEFINE_integer('input_dim', 784, 'size of the input')
tf.app.flags.DEFINE_integer('output_dim', 784, 'size of the output')

# ===== Training Setup=======
tf.app.flags.DEFINE_integer('number_of_epochs', 200, 'number_of_epochs')
tf.app.flags.DEFINE_integer('training_size', 20, 'training_size')
tf.app.flags.DEFINE_float('p_keep_input', 0.9, 'positive float')
tf.app.flags.DEFINE_float('p_keep_hidden', 0.6, 'positive float')

tf.app.flags.DEFINE_float('l2_factor', 0, 'positive float')


def prepare(FLAGS):
  tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want

  # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
  encoder_decoder_dir = os.path.join(FLAGS.log_root,FLAGS.fold_id, 'subject_' + FLAGS.subject_id)
  FLAGS.log_root = os.path.join(FLAGS.log_root,FLAGS.fold_id, 'subject_' + FLAGS.subject_id, FLAGS.direction, FLAGS.model,
                                FLAGS.mapper, FLAGS.exp_name, )
  if not os.path.exists(FLAGS.log_root):
    os.makedirs(FLAGS.log_root)

  test_embeddings, test_normalized_brain_scans, test_words, \
  train_embeddings, train_normalized_brain_scans, train_size, train_words = load_data(FLAGS)
  print("feature flag:", FLAGS.features)
  if FLAGS.features == "dim_reducted":
    with tf.Session() as sess:
      # Restore the model from last checkpoints
      dir = os.path.join(encoder_decoder_dir, "BrainAutoEncoder/detrended_standardized/best/")
      saver = tf.train.import_meta_graph(dir + 'best_model_epoch19.meta')
      saver.restore(sess, tf.train.latest_checkpoint(dir))

      graph = tf.get_default_graph()
      w_in = sess.run(graph.get_tensor_by_name("encoder_layer/w_in:0"))
      w_out = sess.run(graph.get_tensor_by_name("decoder_layaer/w_out:0"))
      b_in = sess.run(graph.get_tensor_by_name("encoder_layer/b_in:0"))
      b_out = sess.run(graph.get_tensor_by_name("decoder_layaer/b_out:0"))

      # Now, access the op that you want to run.
      train_normalized_brain_scans = np.matmul(train_normalized_brain_scans, w_in) + b_in
      test_normalized_brain_scans = np.matmul(test_normalized_brain_scans, w_in) + b_in

      print("shape after reduction:",train_normalized_brain_scans.shape)

  hps = compile_params(train_embeddings, train_normalized_brain_scans, train_size)

  return hps, test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_words


def compile_params(train_embeddings, train_normalized_brain_scans, train_size):
  if FLAGS.direction == "word2brain":
    FLAGS.input_dim = train_embeddings.shape[1]
    FLAGS.output_dim = train_normalized_brain_scans.shape[1]
  elif FLAGS.direction == "brain2word":
    FLAGS.output_dim = train_embeddings.shape[1]
    FLAGS.input_dim = train_normalized_brain_scans.shape[1]
    print("input dim:", FLAGS.input_dim)

  FLAGS.training_size = train_size
  hparam_list = ['batch_size', 'hidden_dim', 'input_dim', 'output_dim', 'number_of_epochs', 'training_size', 'mode',
                 'l2_factor']

  hps_dict = {}
  for key, val in FLAGS.__flags.items():  # for each flag
    if key in hparam_list:  # if it's in the list
      hps_dict[key] = val.value  # add it to the dict

  hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
  return hps


def main(unused_argv):
  hps, test_embeddings, test_normalized_brain_scans, test_words, \
  train_embeddings, train_normalized_brain_scans, train_words = prepare(FLAGS)

  if FLAGS.direction == "brain2word":
    test_embeddings = np.tanh(test_embeddings)
    train_embeddings = np.tanh(train_embeddings)

  if FLAGS.direction == "word2brain":
    from train import train

    if FLAGS.mapper == "intended":
      from VanillaIntendedMapper import VanillaIntendedMapper as StateMapper
    elif FLAGS.mapper == "decoder":
      from DecodedMapper import DecodedMapper as StateMapper
    else:
      from StateMapper import StateMapper as StateMapper
  elif FLAGS.direction == "brain2word":
    from train_Brain2Word import train
    if FLAGS.mapper == "intended":
      from VanillaIntendedMapper_Brain2Word import VanillaIntendedMapper as StateMapper
    elif FLAGS.mapper == "decoder":
      from DecodedMapper import DecodedMapper as StateMapper
    else:
      from StateMapper import StateMapper as StateMapper

  with tf.Graph().as_default():
    # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
    train_dir = os.path.join(FLAGS.log_root, "train")
    best_dir = os.path.join(FLAGS.log_root, "best")
    if not os.path.exists(train_dir): os.makedirs(train_dir)
    if not os.path.exists(best_dir): os.makedirs(best_dir)

    mapper = StateMapper(hps)
    mapper.build_mapping_model()

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
      if FLAGS.direction == "word2brain":
        if FLAGS.mode == "train":
          print("number of train and test examples:",train_normalized_brain_scans.shape,len(test_normalized_brain_scans))
          train(mapper, sess, sv, train_embeddings, train_normalized_brain_scans,
                test_embeddings, test_normalized_brain_scans,
                test_words=test_words, train_words=train_words, FLAGS=FLAGS, best_saver=best_saver, best_dir=best_dir)
        elif FLAGS.mode == "save_vectors":
          print("saving vectors...")
          save_pred_and_target_labesl(mapper, sess, test_embeddings, test_normalized_brain_scans, test_words, FLAGS,
                                      best_saver=best_saver, best_dir=best_dir)
        elif FLAGS.mode == "eval_voxels":
          save_pred_and_target_brain_vectors(mapper, sess, sv, train_embeddings, train_normalized_brain_scans,
                                             test_embeddings, test_normalized_brain_scans,
                                             test_words=test_words, train_words=train_words, FLAGS=FLAGS,
                                             best_saver=best_saver, best_dir=best_dir)


      elif FLAGS.direction == "brain2word":
        if FLAGS.mode == "train":
          print("brain2word training...")
          print("number of train and test examples:",len(train_normalized_brain_scans),len(test_normalized_brain_scans))
          train(mapper, sess, sv, train_normalized_brain_scans, train_embeddings,
                test_normalized_brain_scans, test_embeddings,
                test_words=test_words, train_words=train_words, FLAGS=FLAGS, best_saver=best_saver, best_dir=best_dir)
        elif FLAGS.mode == "save_vectors":
          print("saving vectors...")
          save_pred_and_target_labesl(mapper, sess, test_normalized_brain_scans, test_embeddings, test_words, FLAGS,
                                      best_saver=best_saver, best_dir=best_dir)
        elif FLAGS.mode == "eval_voxels":
          save_pred_and_target_brain_vectors(mapper, sess, sv, train_normalized_brain_scans, train_embeddings,
                                             test_normalized_brain_scans, test_embeddings,
                                             test_words=test_words, train_words=train_words, FLAGS=FLAGS,
                                             best_saver=best_saver, best_dir=best_dir)


if __name__ == '__main__':
  tf.app.run()

