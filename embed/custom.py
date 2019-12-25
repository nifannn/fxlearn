import numpy as np
import random
import tensorflow as tf
import sys
import time
import os
import json

sys.path.append("../")
import util

TF_CONFIG = tf.ConfigProto(allow_soft_placement = True)

class CustomEmbedding(object):
    """docstring for CustomEmbedding"""
    def __init__(self, arg):
        self.arg = arg


    def _build_graph(self):
    	tf.reset_default_graph()
    	self._input_item_1 = tf.placeholder(tf.int32, [None], name='input_item_1')
    	self._input_item_2 = tf.placeholder(tf.int32, [None], name='input_item_2')
    	self._label = tf.placeholder(tf.int32, [None], name='label')
    	self._embed_matrix = tf.Variable(tf.random_uniform((len(self._ix2item), self._embed_dim), -1, 1))
    	
        