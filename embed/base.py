import numpy as np
import tensorflow as tf
import logging
import time

TF_CONFIG = tf.ConfigProto(allow_soft_placement = True)

class BaseEmbedding(object):
    """BaseEmbedding"""
    def __init__(self, arg):
        self.arg = arg

    def get(self):
        pass
        

class CustomEmbedding(object):
	"""docstring for CustomEmbedding"""
	def __init__(self, arg):
		super(CustomEmbedding, self).__init__()
		self.arg = arg
		