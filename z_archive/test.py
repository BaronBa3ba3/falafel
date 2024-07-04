import os
import subprocess

import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.python.keras
# import keras



'''

#### Test 1


print('\n')

print(tf.__version__)

print(tensorflow.keras.__version__)

print('\n')



#### Test 2


path = "".join(["models/", "model_CatDog", ".keras"])

check1 =  os.path.isfile(path)

check2 = os.path.exists(path)

print('\n {} \n {} \n {} \n'.format(path, check1, check2))



#### Test 3
images = [0]*9


plt.figure(figsize=(6,3))

for i in range(len(images)) :
    try:
        plt.subplot((len(images))/2, (len(images))/2, i)
    except:
        plt.subplot((len(images)+1)/2, (len(images)-1)/2, i)


plt.show()


'''


#### Test 4

print(2//4)

print(4/4)