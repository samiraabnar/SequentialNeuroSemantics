import tensorflow as tf
import random
import numpy as np
import matplotlib.pyplot as plt



class Scan(object):
    def __init__(self, activations, timestamp, step, word=None, prev_word=None, next_word=None):
        self.activations = activations
        self.timestamp = timestamp
        self.word = word
        self.prev_word = prev_word
        self.next_word = next_word
        self.step = step



class HPReader(object):
    def __init__(data_dir,datafile):
        pass





