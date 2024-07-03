import os
import subprocess



import tensorflow as tf
import tensorflow.python.keras
# import keras


print('\n')

print(tf.__version__)

print(tensorflow.keras.__version__)

print('\n')


path = "".join(["models/", "model_CatDog", ".keras"])

check1 =  os.path.isfile(path)

check2 = os.path.exists(path)

print('\n {} \n {} \n {} \n'.format(path, check1, check2))