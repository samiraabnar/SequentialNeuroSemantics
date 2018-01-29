import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from  scipy.stats import pearsonr

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import *

class Scan(object):
    def __init__(self, activations, timestamp, step, word=None, prev_word=None, next_word=None):
        self.activations = activations
        self.timestamp = timestamp
        self.word = word
        self.prev_word = prev_word
        self.next_word = next_word
        self.step = step





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

    selected_indices = select_best_features(brain_scans,words)


    min_voxel_value = np.min(brain_scans)
    max_voxel_value = np.max(brain_scans)
    print("brain scans min max: %f %f" % (min_voxel_value, max_voxel_value))
    normalized_brain_scans = (brain_scans - min_voxel_value) / (max_voxel_value - min_voxel_value)
    nmin_voxel_value = np.min(normalized_brain_scans)
    nmax_voxel_value = np.max(normalized_brain_scans)
    print("normalized brain scans min max: %f %f" % (nmin_voxel_value, nmax_voxel_value))
    #print(len(normalized_brain_scans))
    return lstm_embeddings, normalized_brain_scans[selected_indices], words


def read_and_prepare_data_block_based(block_ids,layer_id):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    embeddings = np.load("../data/lstm_"+str(layer_id)+"_emb_objects.npy")
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

    selected_indices = select_best_features(brain_scans,current_word)

    #print(len(normalized_brain_scans))
    return lstm_embeddings, brain_scans[:,selected_indices], words


#words, word embeddings, associated brain scans
def read_and_prepare_data_word_based(block_ids,word_embedding_dic_file="../data/word_embedding_dic.npy"):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    word_embeddings = np.load(word_embedding_dic_file)
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    brain_scans = []
    brain_scan_steps = []
    scan_words = []

    embeddings = []
    all_scans = []
    all_scanned_words = []
    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            all_scans.append(scan_obj.activations[0])
            all_scanned_words.append(scan_obj.word)
            if scan_obj.word in word_embeddings.item().keys():
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                scan_words.append(scan_obj.word)
                embeddings.append(word_embeddings.item()[scan_obj.word])


    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    embeddings = np.asarray(embeddings)
    scan_words = np.asarray(scan_words)

    all_scans = np.asarray(all_scans)
    all_scanned_words = np.asarray(all_scanned_words)

    selected_indices = select_best_features(all_scans,all_scanned_words)

    """
    min_voxel_value = np.min(brain_scans)
    max_voxel_value = np.max(brain_scans)
    print("brain scans min max: %f %f" % (min_voxel_value, max_voxel_value))
    normalized_brain_scans = (brain_scans - min_voxel_value) / (max_voxel_value - min_voxel_value)
    nmin_voxel_value = np.min(normalized_brain_scans)
    nmax_voxel_value = np.max(normalized_brain_scans)
    print("normalized brain scans min max: %f %f" % (nmin_voxel_value, nmax_voxel_value))
    #print(len(normalized_brain_scans))
    """
    normalized_brain_scans = brain_scans # normalize(brain_scans,'l2')

    # min_embedding_value = np.min(embeddings)
    # max_embedding_value = np.max(embeddings)
    normalized_embeddings = embeddings # normalize(embeddings, 'l2')
    # (embeddings - min_embedding_value) / (max_embedding_value - min_embedding_value)


    word_to_scans = {}
    word_to_embeddings = {}
    for i in np.arange(len(normalized_brain_scans)):
        if scan_words[i] not in word_to_scans.keys():
            word_to_scans[scan_words[i]] = []
            word_to_embeddings[scan_words[i]] = []

        word_to_scans[scan_words[i]].append(normalized_brain_scans[i])
        word_to_embeddings[scan_words[i]].append(normalized_embeddings[i])

    words = []
    avg_normalized_scans = []
    avg_word_embeddings = []
    for w in word_to_scans.keys():
        avg_normalized_scans.append(np.mean(word_to_scans[w],axis=0))
        avg_word_embeddings.append(np.mean(word_to_embeddings[w],axis=0))

        words.append(w)

        #print(np.mean(word_to_scans[w],axis=0).shape)




    return np.asarray(avg_word_embeddings), np.asarray(avg_normalized_scans)[:,selected_indices], np.asarray(words)


def load_data(FLAGS):
    if FLAGS.model == "contextual_1":
        train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based([1, 2, 3],layer_id=1)
        test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based([4],layer_id=1)
        train_size = len(train_embeddings)
    elif FLAGS.model == "contextual_0":
        train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based([1, 2, 3],layer_id=0)
        test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based([4],layer_id=0)
        train_size = len(train_embeddings)
    elif FLAGS.model == "char_word":
        test_embeddings, test_normalized_brain_scans, test_words, \
        train_embeddings, train_normalized_brain_scans, train_size, train_words = \
            prepare_trainings_for_character_based_word_embeddings()
    elif FLAGS.model == "word":
        test_embeddings, test_normalized_brain_scans, test_words, \
         train_embeddings, train_normalized_brain_scans, train_size, train_words = \
             prepare_trainings_for_softmax_word_embeddings()
    elif FLAGS.model == "glove":
        test_embeddings, test_normalized_brain_scans, test_words, \
         train_embeddings, train_normalized_brain_scans, train_size, train_words = \
             prepare_trainings_for_glove_word_embeddings()

    print("size of brain scans:",train_normalized_brain_scans.shape)

    return test_embeddings, test_normalized_brain_scans, test_words, \
               train_embeddings, train_normalized_brain_scans, train_size, train_words


def prepare_trainings_for_character_based_word_embeddings():
    word_embeddings, normalized_brain_scans, words = read_and_prepare_data_word_based([1, 2, 3, 4])
    indexes = np.arange(len(normalized_brain_scans))
    random.shuffle(indexes)
    train_size = (len(indexes) // 4) * 3
    train_indexes = indexes[: train_size]
    test_indexes = indexes[train_size:]
    train_embeddings, train_normalized_brain_scans, train_words = \
        word_embeddings[train_indexes], normalized_brain_scans[train_indexes], words[train_indexes]
    test_embeddings, test_normalized_brain_scans, test_words = \
        word_embeddings[test_indexes], normalized_brain_scans[test_indexes], words[test_indexes]
    return test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_size, train_words



def prepare_trainings_for_softmax_word_embeddings():
     word_embeddings, normalized_brain_scans, words = read_and_prepare_data_word_based([1, 2, 3, 4],"../data/harry_softmax_embeddings.npy")
     indexes = np.arange(len(normalized_brain_scans))
     random.shuffle(indexes)
     train_size = (len(indexes) // 4) * 3
     train_indexes = indexes[: train_size]
     test_indexes = indexes[train_size:]
     train_embeddings, train_normalized_brain_scans, train_words = word_embeddings[train_indexes], normalized_brain_scans[train_indexes], words[train_indexes]
     test_embeddings, test_normalized_brain_scans, test_words = \
         word_embeddings[test_indexes], normalized_brain_scans[test_indexes], words[test_indexes]
     return test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_size, train_words    



def prepare_trainings_for_glove_word_embeddings():
     word_embeddings, normalized_brain_scans, words = read_and_prepare_data_word_based([1, 2, 3, 4],"../data/glove_word_embedding_dic.npy")
     indexes = np.arange(len(normalized_brain_scans))
     random.shuffle(indexes)
     train_size = (len(indexes) // 4) * 3
     train_indexes = indexes[: train_size]
     test_indexes = indexes[train_size:]
     train_embeddings, train_normalized_brain_scans, train_words = word_embeddings[train_indexes], normalized_brain_scans[train_indexes], words[train_indexes]
     test_embeddings, test_normalized_brain_scans, test_words = \
         word_embeddings[test_indexes], normalized_brain_scans[test_indexes], words[test_indexes]
     return test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_size, train_words


def select_best_features(brain_scans,scan_words):
    scan_words_set = list(set(scan_words))
    words_ids = [scan_words_set.index(word) for word in scan_words]
    indexes = SelectKBest(f_regression, k=5000).fit(brain_scans,words_ids).get_support(indices=True)

    return indexes

