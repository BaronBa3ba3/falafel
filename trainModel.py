import os
import warnings

import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers
from keras import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16

import getDatabase
# from __main__ import *




def main(directories):

#### Declaring Variables

    train_dir = directories[0]
    validation_dir = directories[1]

    EPOCHS = 3
    BATCH_SIZE = 20  # Number of training examples to process before updating our models variables
    IMG_SHAPE  = 224 # Our training data consists of images with width of 224 pixels and height of 224 pixels
    


#### Augmenting images

    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator( rescale = 1.0/255. )



#### Training and Validation Sets

    # Flow training images in batches of 20 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(train_dir, batch_size = BATCH_SIZE, class_mode = 'binary', target_size = (IMG_SHAPE, IMG_SHAPE))

    # Flow validation images in batches of 20 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = BATCH_SIZE, class_mode = 'binary', target_size = (IMG_SHAPE, IMG_SHAPE))



#### Loading the Base Model

    base_model = VGG16(input_shape = (IMG_SHAPE, IMG_SHAPE, 3), # Shape of our images
    include_top = False, # Leave out the last fully connected layer
    weights = 'imagenet')


    for layer in base_model.layers:
        layer.trainable = False



#### Compiling and Fitting the Model

    # Flatten the output layer to 1 dimension
    x = layers.Flatten()(base_model.output)

    # Add a fully connected layer with 512 hidden units and ReLU activation
    x = layers.Dense(512, activation='relu')(x)

    # Add a dropout rate of 0.5
    x = layers.Dropout(0.5)(x)

    # Add a final sigmoid layer with 1 node for classification output
    x = layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.models.Model(base_model.input, x)

    model.compile(optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

    print("\nModel Summary:")
    model.summary()

    history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 10, epochs = 3)


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
    # plt.savefig('./foo.png')
    plt.show()




#### Saving the Model

    ## Save model using tensorflow.keras
    # model.save('model.pb')
    # model.save('model.h5')

    ## Save model using tensorflow
    # tf.saved_model.save(model, "my_model")



if __name__ == "__main__":
    main()