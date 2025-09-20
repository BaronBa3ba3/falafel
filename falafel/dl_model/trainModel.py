import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf
# from tensorflow.python.keras import layers
from keras.applications.vgg16 import VGG16

# for for TensorFlow < 2.12 (Windows) :
# from keras.preprocessing.image import ImageDataGenerator

# for TensorFlow => 2.16 (WSL/Linux/Docker) : 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


import constants
# from __main__ import *




def main():

#### Declaring Variables

    train_dir = constants.TRAIN_DIR
    validation_dir = constants.VALIDATION_DIR

    EPOCHS = constants.EPOCHS
    BATCH_SIZE = constants.BATCH_SIZE  # Number of training examples to process before updating our models variables
    IMG_SHAPE  = constants.IMG_SHAPE # Our training data consists of images with width of 224 pixels and height of 224 pixels


    total_train = sum([len(files) for r, d, files in os.walk(train_dir)]) # Number of Training Images

    is_training = True
    total_epochs = EPOCHS
    current_epoch = 0



    # Number of Steps per Epoch
    if (constants.N_STEPS_PER_EPOCH == 0):
        n_steps_epoch = int(np.ceil(total_train / float(BATCH_SIZE)))
    else:
        n_steps_epoch = constants.N_STEPS_PER_EPOCH
    
#### Logging

    log_dir = constants.LOG_DIR


#### Augmenting images

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator( rescale = 1.0/255. )



#### Training and Validation Sets

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir, batch_size = BATCH_SIZE, class_mode = 'binary', target_size = (IMG_SHAPE, IMG_SHAPE), subset='training')

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = BATCH_SIZE, class_mode = 'binary', target_size = (IMG_SHAPE, IMG_SHAPE), subset='validation')



#### Loading the Model

    model = tf.keras.models.load_model(constants.MODEL_PATH)




    print("\nModel Summary:\n")
    model.summary()

    history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = n_steps_epoch, epochs = EPOCHS)



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
    plt.savefig(os.path.join(log_dir, 'plots', 'Training_Validation_graphs.png'))

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