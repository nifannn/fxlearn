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

class AirbnbEmbedding(object):
    """docstring for AirbnbEmbedding"""
    def __init__(self, arg):
        self.arg = arg
        
        
        