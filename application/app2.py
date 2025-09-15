from flask import Flask, request, jsonify
from keras.models import load_model
import keras.utils as image
import numpy as np
from PIL import Image
import io
import logging

import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

IMG_SHAPE  = 224 # Our training data consists of images with width of 224 pixels and height of 224 pixels
MODEL_DIR = "/mnt/c/Users/bruno/Documents/1_Programming/z-temp/Models"
MODEL_NAME = "model_CatDog.keras"

modelPath = os.path.join(MODEL_DIR, MODEL_NAME)

# Load the model
model = load_model(modelPath)
@app.route('/predict', methods=['POST'])
def predict():
    app.logger.info('Received a request to /predict')
    app.logger.debug(f'Request headers: {request.headers}')
    app.logger.debug(f'Request files: {request.files}')

    if 'file' not in request.files:
        app.logger.error('No file part in the request')
        return jsonify({'error': 'No file part in the request. Make sure you are sending a file with the key "file" in the request.'}), 400

    file = request.files['file']
    app.logger.info(f'Received file: {file.filename}')

    if file.filename == '':
        app.logger.error('No file selected for uploading')
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        # Read the file into a byte stream
        file_bytes = file.read()
        
        # Use PIL to open the image
        img = Image.open(io.BytesIO(file_bytes))
        
        # Resize and preprocess the image
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Make prediction
        prediction = model.predict(img_array)
        result = 'Dog' if prediction[0][0] > 0.5 else 'Cat'

        app.logger.info(f'Prediction result: {result}')
        return jsonify({'prediction': result}), 200

    except Exception as e:
        app.logger.error(f'Error during prediction: {str(e)}')
        return jsonify({'error': str(e)}), 500
    


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)



#### Request :
#
# curl -X POST -F "file=@data_test/cat.jpg" http://localhost:5000/predict