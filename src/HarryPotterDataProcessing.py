import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import *
from sklearn.decomposition import *

import nilearn.signal

fold= {}
fold["1"] = {'train':[1,2,3],'test':[4]}
fold["2"] = {'train':[2,3,4],'test':[1]}
fold["3"] = {'train':[1,3,4],'test':[2]}
fold["4"] = {'train':[1,2,4],'test':[3]}

class Scan(object):
  def __init__(self, activations, timestamp, step, prev_words=None, next_words=None, all_words=None):
    self.activations = activations
    self.timestamp = timestamp
    self.prev_words = prev_words
    self.next_words = next_words
    self.step = step
    self.all_words = all_words


def read_and_prepare_data_block_based(block_ids, layer_id, scan_objects, FLAGS):
  embeddings = np.load("../embeddings/subject_" + "1" + "_lstm_" + str(layer_id) + "_emb_objects.npy")
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

        while len(all_embeddings) < 4:
          all_embeddings.append(np.zeros_like(all_embeddings[-1]))
        # print("avg phrase emb shape:",np.mean(all_embeddings,axis=0).shape)
        if FLAGS.ith_word >= 0:
          lstm_embeddings.append(all_embeddings[FLAGS.ith_word])
        else:
          lstm_embeddings.append(np.mean(all_embeddings, axis=0))
        words.append('_'.join(scan_obj.all_words))

  all_brain_scans = np.asarray(all_brain_scans)
  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  current_word = np.asarray(current_word)
  lstm_embeddings = np.asarray(lstm_embeddings)
  words = np.asarray(words)

  return lstm_embeddings, brain_scans, words


def read_and_prepare_data_block_based_concat(block_ids, scan_objects, FLAGS):
  embeddings_0 = np.load("../embeddings/subject_" + "1" + "_lstm_" + str(0) + "_emb_objects.npy")
  embeddings_1 = np.load("../embeddings/subject_" + "1" + "_lstm_" + str(1) + "_emb_objects.npy")
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
        for state0, state1 in zip(embeddings_0.item().get(block_id)[scan_obj.step],
                                  embeddings_1.item().get(block_id)[scan_obj.step]):
          state = np.concatenate([state0, state1], axis=0)
          all_embeddings.append(state)

        lstm_embeddings.append(np.mean(all_embeddings, axis=0))
        words.append('_'.join(scan_obj.all_words))

  all_brain_scans = np.asarray(all_brain_scans)
  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  current_word = np.asarray(current_word)
  lstm_embeddings = np.asarray(lstm_embeddings)
  words = np.asarray(words)

  return lstm_embeddings, brain_scans, words


def read_and_prepare_data_block_based_ith_word(block_ids, scan_objects, FLAGS):
  # print(len(scan_objects.item().get(1)))
  # print(embeddings.item().get(1))




  all_brain_scans = []
  brain_scans = []
  brain_scan_steps = []
  current_word = []
  lstm_embeddings = []
  words = []

  for block_id in block_ids:
    layer_id = 0
    lstm_h_0 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_hidden_" + str(layer_id) + ".npy").item()
    lstm_m_0 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_memory_" + str(layer_id) + ".npy").item()

    layer_id = 1
    lstm_h_1 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_hidden_" + str(layer_id) + ".npy").item()
    lstm_m_1 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_memory_" + str(layer_id) + ".npy").item()

    for k  in np.arange(len(scan_objects.item().get(block_id))):
      if (k - FLAGS.ith_step) > 0:
        scan_obj = scan_objects.item().get(block_id)[k - FLAGS.ith_step]
        # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
        if FLAGS.ith_word < len(scan_obj.all_words):
          current_word.append('_'.join(scan_obj.all_words[FLAGS.ith_word]))
          current_block_step = scan_obj.step - scan_objects.item()[block_id][0].step
          print("current block step:",current_block_step)
          brain_scans.append(scan_obj.activations[0])
          all_embeddings = []

          i = FLAGS.ith_word
          one_embeddings = np.concatenate([lstm_h_0[current_block_step+i][0],
                                           lstm_m_0[current_block_step+i][0],
                                           lstm_h_1[current_block_step+i][0],
                                           lstm_m_1[current_block_step+i][0]])


          lstm_embeddings.append(one_embeddings)
          words.append(scan_obj.all_words[FLAGS.ith_word])

  brain_scans = np.asarray(brain_scans)
  lstm_embeddings = np.asarray(lstm_embeddings)
  words = np.asarray(words)

  return lstm_embeddings, brain_scans, words


def read_and_prepare_data_block_based_concat_concat(block_ids, scan_objects, FLAGS):
  # print(len(scan_objects.item().get(1)))
  # print(embeddings.item().get(1))




  all_brain_scans = []
  brain_scans = []
  brain_scan_steps = []
  current_word = []
  lstm_embeddings = []
  words = []

  for block_id in block_ids:
    layer_id = 0
    lstm_h_0 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_hidden_" + str(layer_id) + ".npy").item()
    lstm_m_0 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_memory_" + str(layer_id) + ".npy").item()

    layer_id = 1
    lstm_h_1 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_hidden_" + str(layer_id) + ".npy").item()
    lstm_m_1 = np.load(
      "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_memory_" + str(layer_id) + ".npy").item()

    for scan_obj in scan_objects.item().get(block_id):
      # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
      current_word.append('_'.join(scan_obj.all_words))
      current_block_step = scan_obj.step - scan_objects.item()[block_id][0].step
      print("current block step:",current_block_step)
      brain_scans.append(scan_obj.activations[0])
      all_embeddings = []

      i = 0
      while i < 4 and (current_block_step+i) < len(lstm_h_0.keys()):
        print(lstm_h_0[current_block_step+i].shape)
        print(lstm_h_1[current_block_step + i].shape)
        all_embeddings.append(np.concatenate([lstm_h_0[current_block_step+i][0],lstm_m_0[current_block_step+i][0],lstm_h_1[current_block_step+i][0],lstm_m_1[current_block_step+i][0]]))
        i += 1

      while len(all_embeddings) < 4:
        all_embeddings.append(np.zeros(all_embeddings[-1].shape))

      lstm_embeddings.append(np.concatenate(all_embeddings, axis=0))
      words.append('_'.join(scan_obj.all_words))

  brain_scans = np.asarray(brain_scans)
  lstm_embeddings = np.asarray(lstm_embeddings)
  words = np.asarray(words)

  return lstm_embeddings, brain_scans, words


def read_and_prepare_data_block_based_avg_concat(block_ids, scan_objects, FLAGS):
  embeddings_0 = np.load("../embeddings/subject_" + FLAGS.subject_id + "_lstm_" + str(0) + "_emb_objects.npy")
  embeddings_1 = np.load("../embeddings/subject_" + FLAGS.subject_id + "_lstm_" + str(1) + "_emb_objects.npy")

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
        print("step: ", scan_obj.step)
        for state0, state1 in zip(embeddings_0.item().get(block_id)[scan_obj.step],
                                  embeddings_1.item().get(block_id)[scan_obj.step]):

          state0 = np.asarray(state0)
          state1 = np.asarray(state1)
          print(state0)
          print(state0.shape)
          print(state1.shape)


          print(state0.shape,state1.shape)
          print(scan_obj.all_words)
          state = np.concatenate([state0, state1], axis=0)
          print("state shape:",state.shape)
          all_embeddings.append(state)

        while len(all_embeddings) < 4:
          all_embeddings.append(np.zeros(all_embeddings[-1].shape))
        # print("avg phrase emb shape:",np.mean(all_embeddings,axis=0).shape)
        lstm_embeddings.append(np.mean(all_embeddings, axis=0))
        words.append('_'.join(scan_obj.all_words))

  all_brain_scans = np.asarray(all_brain_scans)
  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  current_word = np.asarray(current_word)
  lstm_embeddings = np.asarray(lstm_embeddings)
  words = np.asarray(words)

  return lstm_embeddings, brain_scans, words

def read_and_prepare_data_block_based_brain_scans(block_ids, scan_objects, FLAGS):

  # print(len(scan_objects.item().get(1)))
  # print(embeddings.item().get(1))

  all_brain_scans = []
  brain_scans = []
  brain_scan_steps = []
  current_word = []
  words = []

  for block_id in block_ids:
    for scan_obj in scan_objects.item().get(block_id):
      # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
      words.append('_'.join(scan_obj.all_words))
      brain_scans.append(scan_obj.activations[0])
      brain_scan_steps.append(scan_obj.step)

  brain_scans = np.asarray(brain_scans)
  words = np.asarray(words)

  print("brain scans shape:", brain_scans.shape)
  print("words shape:", words.shape)

  return [], brain_scans, words

def prepare_linear(block_ids, embeddings_file, steps, scan_objects, avg=False,one_step=False):
  print("##### prepare linear #####")
  embeddings = np.load(embeddings_file)
  #embeddings.item()[""] = np.zeros_like(embeddings.item()[list(embeddings.item().keys())[0]])
  #embeddings.item()["+"] = np.zeros_like(embeddings.item()[list(embeddings.item().keys())[0]])
  # print(len(scan_objects.item().get(1)))
  # print(embeddings.item().get(1))
  all_brain_scans = []
  brain_scans = []
  brain_scan_steps = []
  current_word = []
  word_embeddings = []
  words = []

  for block_id in block_ids:
    for scan_obj in scan_objects.item().get(block_id):
      # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
      all_brain_scans.append(scan_obj.activations[0])
      brain_scans.append(scan_obj.activations[0])
      brain_scan_steps.append(scan_obj.step)
      all_embeddings = []

      for ww in scan_obj.current_translated_words:
        one_embeddings = []
        for word in ww:
          one_embeddings.append(embeddings.item()[word])
        print("len one", len(one_embeddings),np.mean(one_embeddings,axis=0).shape)

        if len(one_embeddings) > 0:
          all_embeddings.append(np.mean(one_embeddings,axis=0))

      print("step embedding length:",len(all_embeddings))

      while len(all_embeddings) < 4:
        all_embeddings.append(np.zeros(all_embeddings[-1].shape))

      print(np.asarray(all_embeddings).shape,np.mean(all_embeddings,axis=0).shape)
      if avg == True:
        word_embeddings.append(np.mean(all_embeddings, axis=0))
      else:
        word_embeddings.extend(all_embeddings)

      words.append('_'.join(scan_obj.all_words))

  if avg == False:
    fine_steps = steps * 4
  else:
    fine_steps = steps

  for i in np.arange(fine_steps - 1):
    word_embeddings.insert(0, np.zeros_like(word_embeddings[0]))

  word_embeddings = np.asarray(word_embeddings)
  print("word em shape:",word_embeddings.shape)
  combined_word_embeddings = []

  if avg == False:
    if one_step == True:
      for i in np.arange(4):
        combined_word_embeddings.append(word_embeddings[np.arange(i,len(word_embeddings) - (fine_steps - i) + 1,4)])
    else:
      for i in np.arange(fine_steps):
        combined_word_embeddings.append(word_embeddings[np.arange(i,len(word_embeddings) - (fine_steps - i) + 1,4)])
  else:
    if one_step == True:
      for i in np.arange(1):
        combined_word_embeddings.append(word_embeddings[np.arange(i,len(word_embeddings) - (fine_steps - i) + 1,1)])
    else:
      for i in np.arange(fine_steps):
        combined_word_embeddings.append(word_embeddings[np.arange(i,len(word_embeddings) - (fine_steps - i) + 1,1)])

  all_brain_scans = np.asarray(all_brain_scans)
  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  word_embeddings = np.asarray(word_embeddings)
  words = np.asarray(words)
  combined_word_embeddings = np.asarray(combined_word_embeddings)

  print("combined shape:", combined_word_embeddings.shape, all_brain_scans.shape)
  return combined_word_embeddings, brain_scans, words



def prepare_linear_lstm(block_ids, steps, scan_objects, avg=False, one_step=False):
  print("##### prepare linear LSTM#####")
  all_brain_scans = []
  brain_scans = []
  brain_scan_steps = []
  current_word = []
  word_embeddings = []
  words = []

  for block_id in block_ids:
    layer_id = 0
    lstm_h_0 = np.load(
        "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_hidden_" + str(layer_id) + ".npy").item()
    lstm_m_0 = np.load(
        "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_memory_" + str(layer_id) + ".npy").item()

    layer_id = 1
    lstm_h_1 = np.load(
        "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_hidden_" + str(layer_id) + ".npy").item()
    lstm_m_1 = np.load(
        "../../lm1b/text_input_full_state/block_" + str(block_id) + "/lstm_memory_" + str(layer_id) + ".npy").item()


      for scan_obj in scan_objects.item().get(block_id):
      # print(scan_obj.step, scan_obj.word, scan_obj.timestamp)
      brain_scans.append(scan_obj.activations[0])
      all_words = []
      brain_scan_steps.append(scan_obj.step)

      current_step = scan_obj.step - scan_objects.item().get(block_id)[0].step
      words.append('_'.join(scan_obj.all_words))
      current_embeddings = []
      i = 0
      while i < 4 and (current_step+i) < len(lstm_h_0) and i < len(scan_obj.all_words):
        print("unit shape:",np.concatenate([lstm_h_0[current_step+i][0],lstm_m_0[current_step+i][0],lstm_h_1[current_step+i][0],lstm_m_1[current_step+i][0]]).shape)
        current_embeddings.append(np.concatenate([lstm_h_0[current_step+i][0],lstm_m_0[current_step+i][0],lstm_h_1[current_step+i][0],lstm_m_1[current_step+i][0]]))
        i += 1

      print(current_step,current_step+i,"step embedding shape:",np.asarray(current_embeddings).shape)

      while len(current_embeddings) < 4:
        current_embeddings.append(np.zeros(current_embeddings[-1].shape))
        # print("avg phrase emb shape:",np.mean(all_embeddings,axis=0).shape)
      print(current_step,current_step+i,"extended step embedding length:",np.asarray(current_embeddings).shape)

      if avg == True:
        word_embeddings.append(np.mean(current_embeddings, axis=0))
      else:
        word_embeddings.extend(current_embeddings)

  fine_steps = steps * 4
  for i in np.arange(fine_steps - 1):
    word_embeddings.insert(0, np.zeros_like(word_embeddings[0]))

  combined_word_embeddings = []

  word_embeddings = np.asarray(word_embeddings)
  print("word em shape",word_embeddings.shape)
  if one_step == True:
    for i in np.arange(4):
      print(word_embeddings[np.arange(i,len(word_embeddings) - (fine_steps - i) + 1,4)].shape)
      combined_word_embeddings.append(word_embeddings[np.arange(i,len(word_embeddings) - (fine_steps - i) + 1,4)])
  else:
    for i in np.arange(fine_steps):
      print(i,word_embeddings[np.arange(i, len(word_embeddings) - (fine_steps - i) + 1, 4)].shape)
      combined_word_embeddings.append(word_embeddings[np.arange(i,len(word_embeddings) - (fine_steps - i) + 1,4)])

  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  word_embeddings = np.asarray(word_embeddings)
  words = np.asarray(words)
  combined_word_embeddings = np.asarray(combined_word_embeddings)

  print("combined shape:", combined_word_embeddings.shape, brain_scans.shape)
  return combined_word_embeddings, brain_scans, words


# words, word embeddings, associated brain scans
def read_and_prepare_data_word_based(block_ids, word_embedding_dic_file, scan_objects):
  word_embeddings = np.load(word_embedding_dic_file)
  word_embeddings.item()[""] = np.zeros_like(word_embeddings.item()[list(word_embeddings.item().keys())[0]])
  word_embeddings.item()["+"] = np.zeros_like(word_embeddings.item()[list(word_embeddings.item().keys())[0]])
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

        embeddings.append(np.mean(all_word_embeddings, axis=0))

  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  embeddings = np.asarray(embeddings)
  scan_words = np.asarray(scan_words)

  normalized_brain_scans = brain_scans  # normalize(brain_scans,'l2')

  normalized_embeddings = embeddings  # normalize(embeddings, 'l2')

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
    avg_normalized_scans.append(np.mean(word_to_scans[w], axis=0))
    avg_word_embeddings.append(np.mean(word_to_embeddings[w], axis=0))

    words.append(w)

  return np.asarray(avg_word_embeddings), np.asarray(avg_normalized_scans)[:, :], np.asarray(words)


def read_and_prepare_data_word_based_concat(block_ids, word_embedding_dic_file, scan_objects, ith_step=0, avg=False):
  word_embeddings = np.load(word_embedding_dic_file)
  #word_embeddings.item()[""] = np.zeros_like(word_embeddings.item()[list(word_embeddings.item().keys())[0]])
  #word_embeddings.item()["+"] = np.zeros_like(word_embeddings.item()[list(word_embeddings.item().keys())[0]])

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
      word_index = int(scan_obj.step / 4) + ith_step
      if word_index < len(scan_objects.item()[block_id]) and word_index > 0:
        all_scans.append(scan_obj.activations[0])
        associated_word_scan = scan_objects.item()[block_id][word_index]
        all_words = []
        for ww in associated_word_scan.current_translated_words:
          all_words.extend(ww)

        print(all_words)
        all_scanned_words.append('_'.join(all_words))
        if set(all_words).issubset(word_embeddings.item().keys()):
          brain_scans.append(scan_obj.activations[0])
          brain_scan_steps.append(scan_obj.step)
          scan_words.append('_'.join(all_words))
          all_word_embeddings = []
          for word in all_words:
            print("1 word embedding shape:",word_embeddings.item()[word].shape)
            all_word_embeddings.append(word_embeddings.item()[word])

          while (len(all_word_embeddings) < 4):
            all_word_embeddings.append(np.zeros(all_word_embeddings[-1].shape))

          all_word_embeddings = np.asarray(all_word_embeddings)
          print("all_word_embeddings shape",all_word_embeddings.shape,avg)
          if avg == True:
            embeddings.append(np.mean(all_word_embeddings, axis=0))
            print("averaged all_word_embeddings shape", np.mean(all_word_embeddings, axis=0).shape)
          else:
            embeddings.append(np.concatenate(all_word_embeddings, axis=0))

  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  embeddings = np.asarray(embeddings)
  scan_words = np.asarray(scan_words)
  normalized_brain_scans = brain_scans  # normalize(brain_scans,'l2')

  normalized_embeddings = embeddings  # normalize(embeddings, 'l2')

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
    word_to_embeddings[w] = np.asarray(word_to_embeddings[w])
    print(w,word_to_embeddings[w].shape)
    avg_normalized_scans.append(np.mean(word_to_scans[w], axis=0))
    avg_word_embeddings.append(np.mean(word_to_embeddings[w], axis=0))

    words.append(w)

  print("embedding shape:",np.asarray(avg_word_embeddings).shape)
  return np.asarray(avg_word_embeddings), np.asarray(avg_normalized_scans)[:, :], np.asarray(words)


def read_and_prepare_data_ith_word_based(ith_word, block_ids, word_embedding_dic_file, scan_objects,ith_step):
  word_embeddings = np.load(word_embedding_dic_file)
  word_embeddings.item()[""] = np.zeros_like(word_embeddings.item()[list(word_embeddings.item().keys())[0]])
  word_embeddings.item()["+"] = np.zeros_like(word_embeddings.item()[list(word_embeddings.item().keys())[0]])

  # print(len(scan_objects.item().get(1)))
  # print(embeddings.item().get(1))
  brain_scans = []
  brain_scan_steps = []
  scan_words = []

  embeddings = []
  all_scans = []
  all_scanned_words = []
  for block_id in block_ids:
    for i in np.arange(len(scan_objects.item().get(block_id))):
      scan = scan_objects.item().get(block_id)[i]
      if (i-ith_step) > 0:
        scan_obj = scan_objects.item().get(block_id)[i - ith_step]
        if (ith_word < len(scan_obj.all_words)):
          all_scans.append(scan_obj.activations[0])
          all_scanned_words.append(scan_obj.all_words[ith_word])
          if scan_obj.all_words[ith_word] in word_embeddings.item().keys():
            brain_scans.append(scan_obj.activations[0])
            brain_scan_steps.append(scan_obj.step)
            scan_words.append(scan_obj.all_words[ith_word])
            embeddings.append(word_embeddings.item()[scan_obj.all_words[ith_word]])

  brain_scans = np.asarray(brain_scans)
  brain_scan_steps = np.asarray(brain_scan_steps)
  embeddings = np.asarray(embeddings)
  scan_words = np.asarray(scan_words)
  normalized_brain_scans = brain_scans  # normalize(brain_scans,'l2')

  normalized_embeddings = embeddings  # normalize(embeddings, 'l2')

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
    avg_normalized_scans.append(np.mean(word_to_scans[w], axis=0))
    avg_word_embeddings.append(np.mean(word_to_embeddings[w], axis=0))

    words.append(w)

  return np.asarray(avg_word_embeddings), np.asarray(avg_normalized_scans)[:, :], np.asarray(words)


def load_data(FLAGS):

  print("model:", FLAGS.model)
  scan_objects = np.load(
    "../processed_data/subject_" + FLAGS.subject_id + "/"+FLAGS.fMRI_preprocess_mode+"subject_" + FLAGS.subject_id + "_scan_objects.npy")

  if FLAGS.model == "contextual_1":
    train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based(fold[FLAGS.fold_id]['train'],
                                                                                                    layer_id=1,
                                                                                                    scan_objects=scan_objects,
                                                                                                    FLAGS=FLAGS)
    test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based(fold[FLAGS.fold_id]['test'], layer_id=1,
                                                                                                 scan_objects=scan_objects,
                                                                                                 FLAGS=FLAGS)
    train_size = len(train_embeddings)
  elif FLAGS.model == "contextual_0":
    train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based(fold[FLAGS.fold_id]['train'],
                                                                                                    layer_id=0,
                                                                                                    scan_objects=scan_objects,
                                                                                                    FLAGS=FLAGS)
    test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based(fold[FLAGS.fold_id]['test'], layer_id=0,
                                                                                                 scan_objects=scan_objects,
                                                                                                 FLAGS=FLAGS)
  elif FLAGS.model == "all_lstm":

    if FLAGS.ith_word == -1:
      print("all_lstm")
      train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based_concat_concat(
        fold[FLAGS.fold_id]['train'], scan_objects, FLAGS)
      test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based_concat_concat(fold[FLAGS.fold_id]['test'],scan_objects,FLAGS)

    else:
      train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based_ith_word(
        fold[FLAGS.fold_id]['train'], scan_objects, FLAGS)
      test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based_ith_word(
        fold[FLAGS.fold_id]['test'], scan_objects, FLAGS)

    train_size = len(train_embeddings)
    print("train size: ", train_size)

    train_size = len(train_embeddings)
  elif FLAGS.model == "char_word":
    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = \
      prepare_trainings_for_x_word_embeddings("../embeddings/word_embedding_dic.npy", FLAGS, scan_objects)
  elif FLAGS.model == "word":
    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = \
      prepare_trainings_for_x_word_embeddings(
        "../embeddings/subject_" + FLAGS.subject_id + "_softmax_embedding_dic.npy",
        FLAGS, scan_objects)
  elif FLAGS.model == "glove":
    test_embeddings, test_normalized_brain_scans, test_words = \
      read_and_prepare_data_ith_word_based(int(FLAGS.ith_word), fold[FLAGS.fold_id]['test'],
                                           "../embeddings/filtered_glove_embedding_dic.npy",
                                           scan_objects, FLAGS.ith_step)
    train_embeddings, train_normalized_brain_scans, train_words = \
      read_and_prepare_data_ith_word_based(int(FLAGS.ith_word), fold[FLAGS.fold_id]['train'],"../embeddings/filtered_glove_embedding_dic.npy",
                                           scan_objects,FLAGS.ith_step)

    train_size = len(train_embeddings)
  elif FLAGS.model == "char_word_avg":
    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = \
      prepare_trainings_for_x_word_embeddings("../embeddings/word_embedding_dic.npy", FLAGS, scan_objects, avg=True)
  elif FLAGS.model == "word_avg":
    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = \
      prepare_trainings_for_x_word_embeddings(
        "../embeddings/subject_" + FLAGS.subject_id + "_softmax_embedding_dic.npy",
        FLAGS, scan_objects, avg=True)
  elif FLAGS.model == "glove_avg":
    test_embeddings, test_normalized_brain_scans, test_words, \
    train_embeddings, train_normalized_brain_scans, train_size, train_words = \
      prepare_trainings_for_x_word_embeddings("../embeddings/filtered_glove_embedding_step_dic.npy", FLAGS, scan_objects,
                                              avg=True)
  elif FLAGS.model == "contextual_01":
    print("contextual_01")
    train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based_concat_concat(
      fold[FLAGS.fold_id]['train'], scan_objects, FLAGS)
    test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based_concat_concat(fold[FLAGS.fold_id]['test'],
                                                                                                               scan_objects,
                                                                                                               FLAGS)
    train_size = len(train_embeddings)
    print("train size: ", train_size)
  elif FLAGS.model == "contextual_01_avg":
    print("contextual_01_avg")
    train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based_avg_concat(
      fold[FLAGS.fold_id]['train'], scan_objects, FLAGS)
    test_embeddings, test_normalized_brain_scans, test_words = read_and_prepare_data_block_based_avg_concat(fold[FLAGS.fold_id]['test'],
                                                                                                            scan_objects,
                                                                                                            FLAGS)
    train_size = len(train_embeddings)
    print("train size: ", train_size)
  elif FLAGS.model == "autoencoder":
    train_embeddings, train_normalized_brain_scans, train_words = read_and_prepare_data_block_based_brain_scans(
      [1, 2, 3, 4], scan_objects, FLAGS)
    train_size = len(train_normalized_brain_scans)
    test_embeddings, test_normalized_brain_scans, test_words = [], [], []
    print("train size: ", train_size)
  elif FLAGS.model == "glove_linear":
    print("one step:",FLAGS.one_step)
    train_embeddings, train_normalized_brain_scans, train_words = prepare_linear(fold[FLAGS.fold_id]['train'],
                                                                                 "../embeddings/subject_"+str(FLAGS.subject_id)+"_filtered_glove_embedding_dic.npy",
                                                                                 FLAGS.linear_steps, scan_objects,avg=False,one_step=FLAGS.one_step)
    test_embeddings, test_normalized_brain_scans, test_words = prepare_linear(fold[FLAGS.fold_id]['test'],
                                                                              "../embeddings/subject_"+str(FLAGS.subject_id)+"_filtered_glove_embedding_dic.npy",
                                                                              FLAGS.linear_steps, scan_objects,avg=False,one_step=FLAGS.one_step)
    train_size = train_embeddings.shape[1]
    print("train size: ", train_size)
  elif FLAGS.model == "char_word_linear":
    train_embeddings, train_normalized_brain_scans, train_words = prepare_linear(fold[FLAGS.fold_id]['train'],
                                                                                 "../embeddings/word_embedding_dic.npy",
                                                                                 FLAGS.linear_steps, scan_objects,one_step=FLAGS.one_step)
    test_embeddings, test_normalized_brain_scans, test_words = prepare_linear(fold[FLAGS.fold_id]['test'],
                                                                              "../embeddings/word_embedding_dic.npy",
                                                                              FLAGS.linear_steps, scan_objects,one_step=FLAGS.one_step)
    train_size = train_embeddings.shape[1]
    print("train size: ", train_size)
  elif FLAGS.model == "word_linear":
    train_embeddings, train_normalized_brain_scans, train_words = prepare_linear(fold[FLAGS.fold_id]['train'],
                                                                                 "../embeddings/subject_" + FLAGS.subject_id + "_softmax_embedding_dic.npy",
                                                                                 FLAGS.linear_steps, scan_objects,one_step=FLAGS.one_step)
    test_embeddings, test_normalized_brain_scans, test_words = prepare_linear(fold[FLAGS.fold_id]['test'],
                                                                              "../embeddings/subject_" + FLAGS.subject_id + "_softmax_embedding_dic.npy",
                                                                              FLAGS.linear_steps, scan_objects,one_step=FLAGS.one_step)
    train_size = train_embeddings.shape[1]
    print("train size: ", train_size)
  elif FLAGS.model == "glove_linear_avg":
    train_embeddings, train_normalized_brain_scans, train_words = prepare_linear(fold[FLAGS.fold_id]['train'],
                                                                                 "../embeddings/filtered_glove_embedding_dic.npy",
                                                                                 FLAGS.linear_steps, scan_objects,
                                                                                 avg=True,one_step=FLAGS.one_step)
    test_embeddings, test_normalized_brain_scans, test_words = prepare_linear(fold[FLAGS.fold_id]['test'],
                                                                              "../embeddings/filtered_glove_embedding_dic.npy",
                                                                              FLAGS.linear_steps, scan_objects,
                                                                              avg=True,one_step=FLAGS.one_step)
    train_size = train_embeddings.shape[1]
    print("train size: ", train_size)
  elif FLAGS.model == "char_word_linear_avg":
    train_embeddings, train_normalized_brain_scans, train_words = prepare_linear(fold[FLAGS.fold_id]['train'],
                                                                                 "../embeddings/word_embedding_dic.npy",
                                                                                 FLAGS.linear_steps, scan_objects,
                                                                                 avg=True,one_step=FLAGS.one_step)
    test_embeddings, test_normalized_brain_scans, test_words = prepare_linear(fold[FLAGS.fold_id]['test'],
                                                                              "../embeddings/word_embedding_dic.npy",
                                                                              FLAGS.linear_steps, scan_objects,
                                                                              avg=True,one_step=FLAGS.one_step)
    train_size = train_embeddings.shape[1]
    print("train size: ", train_size)
  elif FLAGS.model == "word_linear_avg":
    train_embeddings, train_normalized_brain_scans, train_words = prepare_linear(fold[FLAGS.fold_id]['train'],
                                                                                 "../embeddings/subject_" + FLAGS.subject_id + "_softmax_embedding_dic.npy",
                                                                                 FLAGS.linear_steps, scan_objects,
                                                                                 avg=True,one_step=FLAGS.one_step)
    test_embeddings, test_normalized_brain_scans, test_words = prepare_linear(fold[FLAGS.fold_id]['test'],
                                                                              "../embeddings/subject_" + FLAGS.subject_id + "_softmax_embedding_dic.npy",
                                                                              FLAGS.linear_steps, scan_objects,
                                                                              avg=True,one_step=FLAGS.one_step)
    train_size = train_embeddings.shape[1]
    print("train size: ", train_size)
  elif FLAGS.model == "lstm_linear":
    train_embeddings, train_normalized_brain_scans, train_words = prepare_linear_lstm(fold[FLAGS.fold_id]['train'],
                                                                                 FLAGS.linear_steps, scan_objects,
                                                                                 avg=False,one_step=FLAGS.one_step)
    test_embeddings, test_normalized_brain_scans, test_words = prepare_linear_lstm(fold[FLAGS.fold_id]['test'],
                                                                                 FLAGS.linear_steps, scan_objects,
                                                                                 avg=False,one_step=FLAGS.one_step)
    train_size = train_embeddings.shape[1]
    print("train size: ", train_size)

  selected_indices = select_best_features(train_normalized_brain_scans, train_words, k=int(FLAGS.select))

  if int(FLAGS.select) > 0:
    train_normalized_brain_scans = train_normalized_brain_scans[:, selected_indices]
    if (len(test_normalized_brain_scans) > 0):
      test_normalized_brain_scans = test_normalized_brain_scans[:, selected_indices]

  print("size of brain scans:", train_normalized_brain_scans.shape)

  if FLAGS.fMRI_preprocess_mode == "nothing":
    train_normalized_brain_scans = (train_normalized_brain_scans - np.min(train_normalized_brain_scans, axis=0)) / (np.max(train_normalized_brain_scans, axis=0)- np.min(train_normalized_brain_scans, axis=0) + 0.000000001)
    test_normalized_brain_scans = (test_normalized_brain_scans - np.min(test_normalized_brain_scans, axis=0)) / (np.max(test_normalized_brain_scans, axis=0)- np.min(test_normalized_brain_scans, axis=0) + 0.000000001)



  print("before:",train_embeddings.shape)
  if len(train_embeddings.shape) > 2:
    all_train_embeddings = train_embeddings.reshape(-1, train_embeddings.shape[-1])
    print("after:",train_embeddings.shape, all_train_embeddings.shape)
  else:
    all_train_embeddings = train_embeddings

  if all_train_embeddings.shape[-1] > 512:
    pca_x = PCA(n_components=512)
    pca_x.fit(np.concatenate([all_train_embeddings],axis=0))

    if len(train_embeddings.shape) > 2:
      new_train_embeddings = []
      for i in np.arange(len(train_embeddings)):
        new_train_embeddings.append(pca_x.transform(train_embeddings[i]))

      new_test_embeddings = []
      for i in np.arange(len(test_embeddings)):
        new_test_embeddings.append(pca_x.transform(test_embeddings[i]))

      train_embeddings = np.asarray(new_train_embeddings)
      test_embeddings = np.asarray(new_test_embeddings)
    else:
      train_embeddings = pca_x.transform(train_embeddings)
      test_embeddings = pca_x.transform(test_embeddings)

  if FLAGS.reduce_brain == True:
    pca_y = PCA(n_components=512)
    print("PCA shape: ", np.concatenate([train_normalized_brain_scans],axis=0).shape)
    pca_y.fit(np.concatenate([train_normalized_brain_scans],axis=0))

    train_normalized_brain_scans = pca_y.transform(train_normalized_brain_scans)
    test_normalized_brain_scans = pca_y.transform(test_normalized_brain_scans)

  print("PCA shape: ", train_normalized_brain_scans.shape)

  return test_embeddings, test_normalized_brain_scans, test_words, \
         train_embeddings, train_normalized_brain_scans, train_size, train_words


def prepare_trainings_for_x_word_embeddings(filename, FLAGS, scan_objects, avg=False):
  if int(FLAGS.ith_word) == -1:
    word_embeddings, normalized_brain_scans, words = read_and_prepare_data_word_based_concat([1, 2, 3, 4], filename,
                                                                                             scan_objects,
                                                                                             FLAGS.ith_step,avg)
  else:
    word_embeddings, normalized_brain_scans, words = read_and_prepare_data_ith_word_based(int(FLAGS.ith_word), [1, 2, 3, 4],
                                                                                          filename, scan_objects,
                                                                                          FLAGS.ith_step)

  indexes = np.arange(len(normalized_brain_scans))
  random.shuffle(indexes)
  train_size = (len(indexes) // 4) * 3
  train_indexes = indexes[: train_size]
  test_indexes = indexes[train_size:]
  train_embeddings, train_normalized_brain_scans, train_words = word_embeddings[train_indexes], normalized_brain_scans[
    train_indexes], words[train_indexes]
  test_embeddings, test_normalized_brain_scans, test_words = \
    word_embeddings[test_indexes], normalized_brain_scans[test_indexes], words[test_indexes]

  return test_embeddings, test_normalized_brain_scans, test_words, train_embeddings, train_normalized_brain_scans, train_size, train_words


def select_best_features(brain_scans, scan_words, k):
  scan_words_set = list(set(scan_words))
  words_ids = [scan_words_set.index(word) for word in scan_words]
  indexes = SelectKBest(f_regression, k=k).fit(brain_scans, words_ids).get_support(indices=True)

  return indexes
