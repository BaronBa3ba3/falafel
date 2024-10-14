import os
import warnings
import time

import matplotlib.pyplot as plt
import numpy as np
import pickle

import tensorflow as tf
# from tensorflow.python.keras import layers
from keras.applications.vgg16 import VGG16


## Do not need ImageDataGenerator (DEPRECATED)
#
# for for TensorFlow < 2.12 (Windows) :
# from keras.preprocessing.image import ImageDataGenerator
#
# for TensorFlow => 2.16 (WSL/Linux/Docker) : 
# from tensorflow.keras.preprocessing.image import ImageDataGenerator


import constants
# from __main__ import *



# Define the normalization function
def normalize(image, label):
	image = tf.cast(image, tf.float32) / 255.0  # Convert pixel values to [0, 1] range
	return image, label


def main():

#### Declaring Variables

	BASE_DATA_DIR = constants.BASE_DATA_DIR
	MODEL_HISTORY_DIR = constants.MODEL_HISTORY_DIR
	LOG_DIR = constants.LOG_DIR


	EPOCHS = constants.EPOCHS
	BATCH_SIZE = constants.BATCH_SIZE  	# Number of training examples to process before updating our models variables
	IMG_SHAPE  = constants.IMG_SHAPE 	# Our training data consists of images with width of 224 pixels and height of 224 pixels
	VAL_SPLIT = constants.VAL_SPLIT

	


	# Number of Steps per Epoch

	# total_train = sum([len(files) for r, d, files in os.walk(train_dir)]) # Number of Training Images

	# if (constants.N_STEPS_PER_EPOCH == 0):
	# 	# n_steps_epoch = int(np.floor(total_train / float(BATCH_SIZE)))
	# 	# n_steps_epoch = total_train // BATCH_SIZE

	# 	n_steps_epoch = int(np.ceil(total_train / float(BATCH_SIZE)))
	# else:
	# 	n_steps_epoch = constants.N_STEPS_PER_EPOCH
	


#### Logging



#### Exporting Dataset (generates an object of type 'tf.data.Dataset' from image files in a directory)

	train_dataset = tf.keras.utils.image_dataset_from_directory(
		BASE_DATA_DIR,
		labels='inferred',
		image_size=(IMG_SHAPE, IMG_SHAPE),  # Resize images to 150x150 (or whatever your model expects)
		batch_size=BATCH_SIZE,
		label_mode='categorical',           # Use 'categorical' for multiclass classification
		validation_split=VAL_SPLIT,         # 15% of data for validation
		subset='training',                  # This specifies to use it for validation
		shuffle=True,                       # Shuffle the dataset
		seed=123,                           # Ensure repeatable shuffling
		verbose=True
	)


	val_dataset = tf.keras.utils.image_dataset_from_directory(
		BASE_DATA_DIR,
		labels='inferred',
		image_size=(IMG_SHAPE, IMG_SHAPE),
		batch_size=BATCH_SIZE,
		label_mode='categorical',
		validation_split=VAL_SPLIT,
		subset='validation',
		shuffle=True,
		seed=123,
		verbose=True
	)


	# Apply normalization to both datasets
	train_dataset = train_dataset.map(normalize)
	val_dataset = val_dataset.map(normalize)

	# Prefetch data for efficiency
	train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
	val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)


#### Loading the Model

	model = tf.keras.models.load_model(constants.MODEL_PATH)

	print("\nModel Summary:\n")
	model.summary()


#### Training

	start_time = time.time()
	
	history = model.fit(train_dataset, validation_data = val_dataset, epochs = EPOCHS)

	# history = model.fit(train_generator, validation_data = validation_generator, epochs = EPOCHS) # method that works best
	# history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = (n_steps_epoch-1), epochs = EPOCHS)

	end_time = time.time()
	duration = end_time - start_time
	print ('\n This Model took %0.2f seconds (%0.1f minutes) to train for %d epochs'%(duration, duration/60, EPOCHS) )



#### Saving the Model History

	nHistories = len([name for name in os.listdir(MODEL_HISTORY_DIR) if os.path.isfile(os.path.join(MODEL_HISTORY_DIR, name))])
	history_file = os.path.join(MODEL_HISTORY_DIR, "".join(['history', str(nHistories), '.pkl']))
	# history_file = os.path.join(MODEL_HISTORY_DIR, 'history.pkl')

	with open(history_file, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)




#### Visualizing results of the training

	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs_range = range(EPOCHS)

	plt.figure(figsize=(8, 8))
	plt.subplot(1, 2, 1)
	plt.plot(epochs_range, acc, label='Training Accuracy')
	plt.plot(epochs_range, val_acc, label='Validation Accuracy')
	plt.legend(loc='lower right')
	plt.title('Training and Validation Accuracy')

	plt.subplot(1, 2, 2)
	plt.plot(epochs_range, loss, label='Training Loss')
	plt.plot(epochs_range, val_loss, label='Validation Loss')
	plt.legend(loc='upper right')
	plt.title('Training and Validation Loss')
	plt.savefig(os.path.join(LOG_DIR, 'plots', 'Training_Validation_graphs.png'))

	plt.close('all')
	# plt.show()




#### Saving the Model

	## Save model using tensorflow.keras
	# model.save('model.pb')
	# model.save('model.h5')

	## Save model using tensorflow
	# tf.saved_model.save(model, "my_model")


	model.save(constants.MODEL_PATH)



if __name__ == "__main__":
	main()