import os

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from keras.utils import load_img, img_to_array
from PIL import Image

import constants





def load_and_preprocess_image_keras(image_path, target_size):
    img = load_img(image_path, target_size=target_size)  # Load image
    img_array = img_to_array(img)  # Convert to array
    img_array = img_array / 255.0  # Normalize to [0,1] range if needed
    return img_array

def load_and_preprocess_image_PIL(image_path, IMG_SHAPE):
    img = Image.open(image_path)
    img = img.resize((IMG_SHAPE, IMG_SHAPE))  # Resize to match model's expected input
    img_array = np.array(img) / 255.0  # Normalize pixel values
    return img_array



def plot_image(i, predictions_array, images):
  prediction, img = predictions_array[i], images[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img)

  predicted_label = "dog" if (round(prediction[0]) == 1) else "cat"

  if predicted_label == "cat":
      prediction = 1 - prediction
  
  plt.xlabel("{} ({:2.4f})%".format(predicted_label,
                                100*np.max(prediction)))



def main():

#### Declaring variables

    IMG_SHAPE = constants.IMG_SHAPE
    modelPath = constants.MODEL_PATH
    predict_dir = constants.PREDICTION_DIR

    # Logging
    log_dir = constants.LOG_DIR
    os.makedirs(os.path.join(log_dir, 'plots'), exist_ok=True)


#### Loading the Model

    model = tf.keras.models.load_model(modelPath)


#### Loading the Images

    image_files = [f for f in os.listdir(predict_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Access the function load_and_preprocess_image for all image files in the directory of images to predict
    images = np.array([load_and_preprocess_image_PIL(os.path.join(predict_dir, img_file), IMG_SHAPE=IMG_SHAPE) for img_file in image_files])
    




#### Prediction

    predictions = model.predict(images)

    print(predictions)



#### Plotting the Predictions

    ## The following plots a figure where the number of supblots is dynamic to the number of images [len(images)] 
    nCols =  5 if (len(images) > 5) else len(images)
    nRows = (len(images) + nCols - 1) // nCols  # This is equivalent to math.ceil(len(images) / nCols)


    fig = plt.figure(figsize=(nCols*4, nRows*4))
    
    for i in range(len(images)) :
        plt.subplot(nRows, nCols, i+1)
        plot_image(i, predictions, images)

    plt.savefig(os.path.join(log_dir, 'plots', 'Predictions.png'))

    plt.close('all')
    # plt.show()


if __name__ == "__main__":
    main()