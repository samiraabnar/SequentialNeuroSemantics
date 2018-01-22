import tensorflow as tf
import numpy as np
import random
from collections import namedtuple
import numpy as np
<<<<<<< HEAD

=======
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import style
>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651

import math
import os

<<<<<<< HEAD
from HarryPotterDataProcessing import *
from train import *


=======
from HarryPotterDataProcessing import Scan
from StateMapper import StateMapper
>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651

FLAGS = tf.app.flags.FLAGS

# ========Where to save outputs===========
tf.app.flags.DEFINE_string('log_root', '../log_root', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('data_path', '../data', 'Directory where the data '
                                                   'is going to be saved.')
<<<<<<< HEAD
tf.app.flags.DEFINE_string('mapper', 'intended', 'intended/forward')
tf.app.flags.DEFINE_string('exp_name', 'primary', 'Name for experiment. Logs will '
                                                          'be saved in a directory with this'
                                                          ' name, under log_root.')
tf.app.flags.DEFINE_string('model', 'contextual_0', 'must be one of '
                                               'char_word/word/contextual_0/contextual_1/contextual_01/char')
=======
tf.app.flags.DEFINE_string('exp_name', 'avg_word_embeddings', 'Name for experiment. Logs will '
                                                          'be saved in a directory with this'
                                                          ' name, under log_root.')
tf.app.flags.DEFINE_string('model', 'vanilla', 'must be one of '
                                               'vanilla')
>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651

# ==========Hyper Params=========
tf.app.flags.DEFINE_integer('batch_size', 10, 'minibatch size')
tf.app.flags.DEFINE_integer('hidden_dim', 1024, 'dimension of hidden states')
tf.app.flags.DEFINE_integer('input_dim', 784, 'size of the input')
tf.app.flags.DEFINE_integer('output_dim', 784, 'size of the output')

# ===== Training Setup=======
tf.app.flags.DEFINE_integer('number_of_epochs', 20, 'number_of_epochs')
tf.app.flags.DEFINE_integer('training_size', 20, 'training_size')


<<<<<<< HEAD



def prepare(FLAGS):
    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want
=======
def read_and_prepare_data():
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    embeddings = np.load("../data/lstm_emb_objects.npy")
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    block_id = 1
    brain_scans = []
    brain_scan_steps = []
    current_word = []
    lstm_embeddings = []
    words = []
    for scan_obj in scan_objects.item().get(block_id):
        # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
        brain_scans.append(scan_obj.activations[0])
        brain_scan_steps.append(scan_obj.step)
        current_word.append(scan_obj.word)
        lstm_embeddings.append(embeddings.item().get(block_id)[scan_obj.step])
        words.append(scan_obj.word)

    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    current_word = np.asarray(current_word)
    lstm_embeddings = np.asarray(lstm_embeddings)
    words = np.asarray(words)

    min_voxel_value = np.min(brain_scans)
    max_voxel_value = np.max(brain_scans)
    print("brain scans min max: %f %f" % (min_voxel_value, max_voxel_value))
    normalized_brain_scans = (brain_scans - min_voxel_value) / (max_voxel_value - min_voxel_value)
    nmin_voxel_value = np.min(normalized_brain_scans)
    nmax_voxel_value = np.max(normalized_brain_scans)
    print("normalized brain scans min max: %f %f" % (nmin_voxel_value, nmax_voxel_value))
    print(len(normalized_brain_scans))
    return lstm_embeddings, normalized_brain_scans, words


def read_and_prepare_data_block_based(block_ids):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    embeddings = np.load("../data/lstm_emb_objects.npy")
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    brain_scans = []
    brain_scan_steps = []
    current_word = []
    lstm_embeddings = []
    words = []

    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
            brain_scans.append(scan_obj.activations[0])
            brain_scan_steps.append(scan_obj.step)
            current_word.append(scan_obj.word)
            if scan_obj.step in embeddings.item().get(block_id).keys():
                lstm_embeddings.append(embeddings.item().get(block_id)[scan_obj.step])
                words.append(scan_obj.word)

    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    current_word = np.asarray(current_word)
    lstm_embeddings = np.asarray(lstm_embeddings)
    words = np.asarray(words)

    min_voxel_value = np.min(brain_scans)
    max_voxel_value = np.max(brain_scans)
    print("brain scans min max: %f %f" % (min_voxel_value, max_voxel_value))
    normalized_brain_scans = (brain_scans - min_voxel_value) / (max_voxel_value - min_voxel_value)
    nmin_voxel_value = np.min(normalized_brain_scans)
    nmax_voxel_value = np.max(normalized_brain_scans)
    print("normalized brain scans min max: %f %f" % (nmin_voxel_value, nmax_voxel_value))
    print(len(normalized_brain_scans))
    return lstm_embeddings, normalized_brain_scans, words


#words, word embeddings, associated brain scans
def read_and_prepare_data_word_based(block_ids):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    word_embeddings = np.load("../data/word_embedding_dic.npy")
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    brain_scans = []
    brain_scan_steps = []
    scan_words = []

    embeddings = []

    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            if scan_obj.word in word_embeddings.item().keys():
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                scan_words.append(scan_obj.word)
                embeddings.append(word_embeddings.item()[scan_obj.word])


    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    embeddings = np.asarray(embeddings)
    scan_words = np.asarray(scan_words)


    min_voxel_value = np.min(brain_scans)
    max_voxel_value = np.max(brain_scans)
    print("brain scans min max: %f %f" % (min_voxel_value, max_voxel_value))
    normalized_brain_scans = (brain_scans - min_voxel_value) / (max_voxel_value - min_voxel_value)
    nmin_voxel_value = np.min(normalized_brain_scans)
    nmax_voxel_value = np.max(normalized_brain_scans)
    print("normalized brain scans min max: %f %f" % (nmin_voxel_value, nmax_voxel_value))
    print(len(normalized_brain_scans))

    word_to_scans = {}
    word_to_embeddings = {}
    for i in np.arange(len(normalized_brain_scans)):
        if scan_words[i] not in word_to_scans.keys():
            word_to_scans[scan_words[i]] = []
            word_to_embeddings[scan_words[i]] = []

        word_to_scans[scan_words[i]].append(normalized_brain_scans[i])
        word_to_embeddings[scan_words[i]].append(embeddings[i])

    words = []
    avg_normalized_scans = []
    avg_word_embeddings = []
    for w in word_to_scans.keys():
        avg_normalized_scans.append(np.mean(word_to_scans[w],axis=0))
        avg_word_embeddings.append(np.mean(word_to_embeddings[w],axis=0))

        words.append(w)

        print(np.mean(word_to_scans[w],axis=0).shape)




    return np.asarray(avg_word_embeddings), np.asarray(avg_normalized_scans), np.asarray(words)


def plot(predictions, targets, train_step, words, plot_size):
    style.use('seaborn')
    font_dict = {'family': 'serif',
                                 'color':'darkred',
                                  'size':5}

    fig = plt.figure(figsize=(plot_size, plot_size))
    gs = gridspec.GridSpec(plot_size, plot_size)
    gs.update(wspace=0.5, hspace=0.5)

    print(len(predictions))
    print(len(targets))
    for i, p_t in enumerate(zip(predictions, targets)):
        p, t = p_t
        ax = plt.subplot(gs[i])
        sorted_voxel_indexes = np.argsort(t)
        time_steps = np.arange(len(sorted_voxel_indexes))
        ax.plot(time_steps, t[sorted_voxel_indexes], time_steps, p[sorted_voxel_indexes], linewidth=0.5)
        ax.axis('off')
        #ax.set_xlabel('voxels (sorted based on label)')
        #ax.set_ylabel('label and predicted')
        ax.set_title(words[i], fontdict=font_dict)
        ax.grid(False)

    plots_path = os.path.join(FLAGS.log_root, 'plots')
    if not os.path.exists(plots_path): os.makedirs(plots_path)

    plt.savefig(plots_path + '/{}.png'.format(str(train_step).zfill(3)),
                bbox_inches='tight', dpi=300)

    tf.logging.info("output plots for test data: " +
                    plots_path + '/{}.png'.format(str(train_step).zfill(3)))

    plt.close(fig)

    return fig


def train(model, sess, sv, train_x, train_y, test_x, test_y, test_words, train_words):
    # Use the session to train the graph.
    training_step = 0
    while not sv.should_stop():
        for i in range(model.hparams.number_of_epochs):

            XY = list(zip(train_x, train_y))
            random.shuffle(XY)
            train_x, train_y = zip(*XY)

            start_index = 0;
            end_index = start_index + model.hparams.batch_size
            while end_index < len(train_x):
                summary, _, cost, mse = sess.run([model.summ_op,model.train_op, model.cost, model.mean_squared_loss],
                                        feed_dict={model.input_states_batch: train_x[start_index:end_index],
                                                   model.output_states_batch: train_y[start_index:end_index],
                                                   model.p_keep_input: 0.8, model.p_keep_hidden: 0.5})
                print("mse %f" % mse)
                start_index = end_index
                end_index = start_index + model.hparams.batch_size
                training_step += 1

            sv.summary_computed(sess, summary)
            qualitative_eval(model, sess, test_x, test_y, training_step,test_words)


def qualitative_eval(model, sess, test_x, test_y, train_step,test_words):
    test_size = 100
    predicted_output = sess.run([model.predicted_output],
                                feed_dict={model.input_states_batch: test_x[:test_size],
                                           model.p_keep_input: 1.0, model.p_keep_hidden: 1.0}
                                )

    print(predicted_output[0].shape)
    plot(predicted_output[0], test_y[:test_size], train_step, test_words[:test_size], int(math.sqrt(test_size)))


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)  # choose what level of logging you want

>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651
    # Change log_root to FLAGS.log_root/FLAGS.exp_name and create the dir if necessary
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.model, FLAGS.exp_name)
    if not os.path.exists(FLAGS.log_root):
        os.makedirs(FLAGS.log_root)
<<<<<<< HEAD
    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = load_data(FLAGS)

    hps = compile_params(train_embeddings, train_normalized_brain_scans, train_size)

    return hps, test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_words

def compile_params(train_embeddings, train_normalized_brain_scans, train_size):
    FLAGS.input_dim = train_embeddings.shape[1]
    FLAGS.output_dim = train_normalized_brain_scans.shape[1]
    FLAGS.training_size = train_size
    hparam_list = ['batch_size', 'hidden_dim', 'input_dim', 'output_dim', 'number_of_epochs', 'training_size']
=======

    """lstm_embeddings, normalized_brain_scans, words= read_and_prepare_data()
    indexes = np.arange(len(normalized_brain_scans))
    random.shuffle(indexes)
    train_size = (len(indexes) // 4) * 3
    print("train_size is %d" % train_size)
    train_indexes = indexes[: train_size]
    test_indexes = indexes[train_size:]
    """

    """train_lstm_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based([1,2,3])
    test_lstm_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based([4])
    train_size = len(train_lstm_embeddings)
    """

    word_embeddings, normalized_brain_scans, words = read_and_prepare_data_word_based([1,2,3,4])
    indexes = np.arange(len(normalized_brain_scans))
    random.shuffle(indexes)
    train_size = (len(indexes) // 4) * 3
    print("train_size is %d" % train_size)
    train_indexes = indexes[: train_size]
    test_indexes = indexes[train_size:]

    train_embeddings,train_normalized_brain_scans, train_words = \
        word_embeddings[train_indexes],normalized_brain_scans[train_indexes], words[train_indexes]
    test_embeddings, test_normalized_brain_scans, test_words = \
        word_embeddings[test_indexes], normalized_brain_scans[test_indexes], words[test_indexes]

    FLAGS.input_dim = train_embeddings.shape[1]
    FLAGS.output_dim = train_normalized_brain_scans.shape[1]
    FLAGS.training_size = train_size

    hparam_list = ['batch_size', 'hidden_dim', 'input_dim', 'output_dim', 'number_of_epochs', 'training_size']

>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651
    hps_dict = {}
    for key, val in FLAGS.__flags.items():  # for each flag
        if key in hparam_list:  # if it's in the list
            hps_dict[key] = val  # add it to the dict
    hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)
<<<<<<< HEAD
    return hps

def main(unused_argv):
    hps, test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_words = prepare(FLAGS)

    if FLAGS.mapper == "intended":
        from VanillaIntendedMapper import VanillaIntendedMapper as StateMapper
    else:
        from StateMapper import StateMapper as StateMapper
=======
>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651

    with tf.Graph().as_default():
        # Create a Supervisor that will checkpoint the model in '/tmp/mydir'.
        train_dir = os.path.join(FLAGS.log_root, "train")
        if not os.path.exists(train_dir): os.makedirs(train_dir)
<<<<<<< HEAD

=======
>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651
        mapper = StateMapper(hps)
        mapper.build_mapping_model()

        saver = tf.train.Saver(max_to_keep=3)
        sv = tf.train.Supervisor(logdir=train_dir,
                                 is_chief=True,
                                 saver=saver,
                                 summary_op=None,
                                 save_summaries_secs=60,  # save summaries for tensorboard every 60 secs
                                 save_model_secs=60,  # checkpoint every 60 secs
<<<<<<< HEAD
                                 )
=======
                                 global_step=None)
>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651
        # Get a TensorFlow session managed by the supervisor.
        with sv.managed_session() as sess:
            train(mapper, sess, sv, train_embeddings, train_normalized_brain_scans,
                  test_embeddings, test_normalized_brain_scans,
<<<<<<< HEAD
                  test_words=test_words,train_words=train_words,FLAGS=FLAGS)

=======
                  test_words=test_words,train_words=train_words)
>>>>>>> 3a3c348a89dc907de1731cea03e8a7d5032aa651


if __name__ == '__main__':
    tf.app.run()
