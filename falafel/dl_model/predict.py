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



def plot_image(i, predicted_percentage, predicted_class, image):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(image)
  
  plt.xlabel("{} ({:2.4f})%".format(predicted_class, 100*(predicted_percentage)))



def main():

#### Declaring variables

    IMG_SHAPE = constants.IMG_SHAPE
    modelPath = constants.MODEL_PATH
    predict_dir = constants.PREDICTION_DIR
    CLASS_LABELS = constants.CLASS_LABELS

    # Logging
    log_dir = constants.LOG_DIR


#### Loading the Model

    model = tf.keras.models.load_model(modelPath)


#### Loading the Images

    image_files = [f for f in os.listdir(predict_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

    # Access the function load_and_preprocess_image for all image files in the directory of images to predict
    images = np.array([load_and_preprocess_image_PIL(os.path.join(predict_dir, img_file), IMG_SHAPE=IMG_SHAPE) for img_file in image_files])
    


#### Prediction (most likely class)

    predictions = model.predict(images)
    print(predictions)

    predicted_classes_index = np.argmax(predictions, axis=1)

    predicted_classes = [CLASS_LABELS[i] for i in predicted_classes_index]
    predicted_values = [predictions[i][predicted_classes_index[i]] for i in range(images.shape[0])]
    # predicted_values = [predictions[predicted_classes_index[i]] for i in range(5)]

    print(predicted_classes)
    print(predicted_values)


#### Prediction (2 most likely classes)

    top_2_indices = np.argsort(predictions, axis=1)[:, -2:]  # Get the last two indices in sorted order

    # Create a list to store the top 2 predicted classes and their probabilities
    # 3 dimension array :       top_2_classes_with_probs = [image1, image2, image3, ...]
    #                           top_2_classes_with_probs[0] = [[class1, prob1], [class2, prob2]]
    top_2_classes_with_probs = []

    # Iterate over each prediction and extract the class labels and probabilities
    for i, indices in enumerate(top_2_indices):
        
        indices = reversed(indices) # Reverse indices to have the highest probability first
        
        # For each index, get the corresponding class label and probability percentage
        top_2_classes_probs = [(CLASS_LABELS[idx], predictions[i][idx]) for idx in indices]
        
        top_2_classes_with_probs.append(top_2_classes_probs)


    # Output the top 2 predicted classes and their probabilities for each image
    for i, classes_probs in enumerate(top_2_classes_with_probs):
        print(f"Image {i+1}:")
        for class_label, prob in classes_probs:
            print(f"  Class: {class_label}, Probability: {prob:.2f}%")


#### Plotting the Predictions

    ## The following plots a figure where the number of supblots is dynamic to the number of images [len(images)] 
    nCols =  5 if (len(images) > 5) else len(images)
    nRows = (len(images) + nCols - 1) // nCols  # This is equivalent to math.ceil(len(images) / nCols)


    fig = plt.figure(figsize=(nCols*4, nRows*4))
    
    for i in range(len(images)) :
        plt.subplot(nRows, nCols, i+1)
        plot_image(i, predicted_values[i], predicted_classes[i], images[i])

    plt.savefig(os.path.join(log_dir, 'plots', 'Predictions.png'))

    plt.close('all')
    # plt.show()


if __name__ == "__main__":
    main()