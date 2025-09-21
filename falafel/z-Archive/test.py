# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pickle

import tensorflow as tf
# from tensorflow.python.keras import layers
from keras.applications.vgg16 import VGG16

# for for TensorFlow < 2.12 (Windows) :
from keras.preprocessing.image import ImageDataGenerator


MODEL_HISTORY_DIR = os.path.join('falafel/dl_model/models', 'model_history/')

nHistories = len([name for name in os.listdir(MODEL_HISTORY_DIR) if os.path.isfile(os.path.join(MODEL_HISTORY_DIR, name))])

history_file = os.path.join(MODEL_HISTORY_DIR, "".join(['history', str(nHistories), '.pkl']))

print (os.listdir(MODEL_HISTORY_DIR))

# history_file = os.path.join(MODEL_HISTORY_DIR, 'history.pkl')
