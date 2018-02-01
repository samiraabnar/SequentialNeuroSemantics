import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from  scipy.stats import pearsonr

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import *

class Scan(object):
    def __init__(self,activations,timestamp, step,prev_words=None,next_words=None,all_words=None):
        self.activations = activations
        self.timestamp = timestamp
        self.prev_words = prev_words
        self.next_words = next_words
        self.step = step
        self.all_words = all_words



def read_and_prepare_data_block_based(block_ids,layer_id):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    embeddings = np.load("../data/lstm_"+str(layer_id)+"_emb_objects.npy")
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    all_brain_scans = []
    brain_scans = []
    brain_scan_steps = []
    current_word = []
    lstm_embeddings = []
    words = []

    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
            all_brain_scans.append(scan_obj.activations[0])
            current_word.append('_'.join(scan_obj.all_words))
            if scan_obj.step in embeddings.item().get(block_id).keys():
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                all_embeddings = []
                for state in embeddings.item().get(block_id)[scan_obj.step]:
                    all_embeddings.append(state)

                print("avg phrase emb shape:",np.mean(all_embeddings,axis=0).shape)
                lstm_embeddings.append(np.mean(all_embeddings,axis=0))
                words.append('_'.join(scan_obj.all_words))

    all_brain_scans = np.asarray(all_brain_scans)
    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    current_word = np.asarray(current_word)
    lstm_embeddings = np.asarray(lstm_embeddings)
    words = np.asarray(words)

    selected_indices = select_best_features(all_brain_scans,current_word)
    brain_scans = normalize(brain_scans[:,selected_indices],'l2')

    return lstm_embeddings, brain_scans, words


def read_and_prepare_data_block_based_concat(block_ids):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    embeddings_0 = np.load("../data/lstm_"+str(0)+"_emb_objects.npy")
    embeddings_1 = np.load("../data/lstm_"+str(1)+"_emb_objects.npy")
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    all_brain_scans = []
    brain_scans = []
    brain_scan_steps = []
    current_word = []
    lstm_embeddings = []
    words = []

    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
            all_brain_scans.append(scan_obj.activations[0])
            current_word.append('_'.join(scan_obj.all_words))
            if scan_obj.step in embeddings_0.item().get(block_id).keys():
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                all_embeddings = []
                for state0,state1 in zip(embeddings_0.item().get(block_id)[scan_obj.step],embeddings_1.item().get(block_id)[scan_obj.step]):
                    state = np.concatenate([state0,state1],axis=0)
                    all_embeddings.append(state)

                print("avg phrase emb shape:",np.mean(all_embeddings,axis=0).shape)
                lstm_embeddings.append(np.mean(all_embeddings,axis=0))
                words.append('_'.join(scan_obj.all_words))

    all_brain_scans = np.asarray(all_brain_scans)
    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    current_word = np.asarray(current_word)
    lstm_embeddings = np.asarray(lstm_embeddings)
    words = np.asarray(words)

    selected_indices = select_best_features(all_brain_scans,current_word)
    brain_scans = normalize(brain_scans[:,selected_indices],'l2')
    #brain_scans = normalize(brain_scans,'l2')
    #print(len(normalized_brain_scans))
    return lstm_embeddings, brain_scans, words


def read_and_prepare_data_block_based_concat_concat(block_ids):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    embeddings_0 = np.load("../data/lstm_"+str(0)+"_emb_objects.npy")
    embeddings_1 = np.load("../data/lstm_"+str(1)+"_emb_objects.npy")
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    all_brain_scans = []
    brain_scans = []
    brain_scan_steps = []
    current_word = []
    lstm_embeddings = []
    words = []

    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
            all_brain_scans.append(scan_obj.activations[0])
            current_word.append('_'.join(scan_obj.all_words))
            if scan_obj.step in embeddings_0.item().get(block_id).keys():
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                all_embeddings = []
                for state0,state1 in zip(embeddings_0.item().get(block_id)[scan_obj.step],embeddings_1.item().get(block_id)[scan_obj.step]):
                    state = np.concatenate([state0,state1],axis=0)
                    all_embeddings.append(state)

                while len(all_embeddings) < 4:
                    all_embeddings.append(np.zeros(all_embeddings[-1].shape))
                print("avg phrase emb shape:",np.concatenate(all_embeddings,axis=0).shape)
                lstm_embeddings.append(np.concatenate(all_embeddings,axis=0))
                words.append('_'.join(scan_obj.all_words))

    all_brain_scans = np.asarray(all_brain_scans)
    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    current_word = np.asarray(current_word)
    lstm_embeddings = np.asarray(lstm_embeddings)
    words = np.asarray(words)

    selected_indices = select_best_features(all_brain_scans,current_word)
    
    #selected_indices = select_best_features(all_brain_scans,current_word)
    brain_scans = normalize(brain_scans[:,selected_indices],'l2')

    print("lstm emb shape:",lstm_embeddings.shape)
    print("brain scans shape:",brain_scans.shape)

    return lstm_embeddings, brain_scans, words


def read_and_prepare_data_block_based_avg_concat(block_ids):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    embeddings_0 = np.load("../data/lstm_"+str(0)+"_emb_objects.npy")
    embeddings_1 = np.load("../data/lstm_"+str(1)+"_emb_objects.npy")
    # print(len(scan_objects.item().get(1)))
    # print(embeddings.item().get(1))
    all_brain_scans = []
    brain_scans = []
    brain_scan_steps = []
    current_word = []
    lstm_embeddings = []
    words = []

    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
            all_brain_scans.append(scan_obj.activations[0])
            current_word.append('_'.join(scan_obj.all_words))
            if scan_obj.step in embeddings_0.item().get(block_id).keys():
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                all_embeddings = []
                for state0,state1 in zip(embeddings_0.item().get(block_id)[scan_obj.step],embeddings_1.item().get(block_id)[scan_obj.step]):
                    state = np.concatenate([state0,state1],axis=0)
                    all_embeddings.append(state)

                while len(all_embeddings) < 4:
                    all_embeddings.append(np.zeros(all_embeddings[-1].shape))
                print("avg phrase emb shape:",np.mean(all_embeddings,axis=0).shape)
                lstm_embeddings.append(np.mean(all_embeddings,axis=0))
                words.append('_'.join(scan_obj.all_words))

    all_brain_scans = np.asarray(all_brain_scans)
    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    current_word = np.asarray(current_word)
    lstm_embeddings = np.asarray(lstm_embeddings)
    words = np.asarray(words)

    selected_indices = select_best_features(all_brain_scans,current_word)
    brain_scans = brain_scans[:,selected_indices]
    print("lstm emb shape:",lstm_embeddings.shape)
    print("brain scans shape:",brain_scans.shape)
    #selected_indices = select_best_features(all_brain_scans,current_word)

    return lstm_embeddings, brain_scans, words

#words, word embeddings, associated brain scans
def read_and_prepare_data_word_based(block_ids,word_embedding_dic_file="../data/word_embedding_dic.npy"):
    scan_objects = np.load("../data/subject_1_scan_objects.npy")
    word_embeddings = np.load(word_embedding_dic_file)
    brain_scans = []
    brain_scan_steps = []
    scan_words = []

    embeddings = []
    all_scans = []
    all_scanned_words = []
    for block_id in block_ids:
        for scan_obj in scan_objects.item().get(block_id):
            all_scans.append(scan_obj.activations[0])
            all_scanned_words.append('_'.join(scan_obj.all_words))
            if set(scan_obj.all_words).issubset(word_embeddings.item().keys()):
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                scan_words.append('_'.join(scan_obj.all_words))
                all_word_embeddings = []
                for word in scan_obj.all_words:
                    all_word_embeddings.append(word_embeddings.item()[word])


                embeddings.append(np.mean(all_word_embeddings,axis=0))


    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    embeddings = np.asarray(embeddings)
    scan_words = np.asarray(scan_words)


    normalized_brain_scans =  brain_scans # normalize(brain_scans,'l2')

    normalized_embeddings = embeddings #normalize(embeddings, 'l2')


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

    return np.asarray(avg_word_embeddings), np.asarray(avg_normalized_scans)[:,:], np.asarray(words)



def read_and_prepare_data_word_based_concat(block_ids,word_embedding_dic_file="../data/word_embedding_dic.npy"):
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
            all_scanned_words.append('_'.join(scan_obj.all_words))
            if set(scan_obj.all_words).issubset(word_embeddings.item().keys()):
                brain_scans.append(scan_obj.activations[0])
                brain_scan_steps.append(scan_obj.step)
                scan_words.append('_'.join(scan_obj.all_words))
                all_word_embeddings = []
                for word in scan_obj.all_words:
                    all_word_embeddings.append(word_embeddings.item()[word])

                while( len(all_word_embeddings) < 4):
                    all_word_embeddings.append(np.zeros(all_word_embeddings[-1].shape))

                embeddings.append(np.concatenate(all_word_embeddings,axis=0))


    brain_scans = np.asarray(brain_scans)
    brain_scan_steps = np.asarray(brain_scan_steps)
    embeddings = np.asarray(embeddings)
    scan_words = np.asarray(scan_words)
    normalized_brain_scans =  brain_scans # normalize(brain_scans,'l2')

    normalized_embeddings = embeddings #normalize(embeddings, 'l2')


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



    return np.asarray(avg_word_embeddings), np.asarray(avg_normalized_scans)[:,:], np.asarray(words)


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
    elif FLAGS.model == "contextual_01":
        print("contextual_01")
        train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based_concat_concat([1, 2, 3])
        test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based_concat_concat([4])
        train_size = len(train_embeddings)
        print("train size: ",train_size)
    elif FLAGS.model == "contextual_01_avg":
        print("contextual_01_avg")
        train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based_avg_concat([1, 2, 3])
        test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based_avg_concat([4])
        train_size = len(train_embeddings)
        print("train size: ",train_size)

    print("size of brain scans:",train_normalized_brain_scans.shape)

    return test_embeddings, test_normalized_brain_scans, test_words, \
               train_embeddings, train_normalized_brain_scans, train_size, train_words


def prepare_trainings_for_character_based_word_embeddings():
    word_embeddings, normalized_brain_scans, words = read_and_prepare_data_word_based([1, 2, 3, 4])
    indexes = np.arange(len(normalized_brain_scans))
    random.shuffle(indexes)
    train_size = (len(indexes) // 5) * 4
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
     word_embeddings, normalized_brain_scans, words = read_and_prepare_data_word_based_concat([1, 2, 3, 4],"../data/glove_word_embedding_dic.npy")
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
    indexes = SelectKBest(f_regression, k=500).fit(brain_scans,words_ids).get_support(indices=True)

    return indexes

