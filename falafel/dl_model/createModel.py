import tensorflow as tf
from keras import layers
from keras.applications.vgg16 import VGG16

import constants



def main():

#### Loading the Base Model

    IMG_SHAPE = constants.IMG_SHAPE
    modelPath = constants.MODEL_PATH

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



#### Saving the Model


    model.save(modelPath)



if __name__ == "__main__":
    main()