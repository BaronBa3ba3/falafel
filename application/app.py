from flask import Flask, request, jsonify
from keras.models import load_model
import keras.utils as image

from PIL import Image
import numpy as np
import io
import logging
from logging.handlers import RotatingFileHandler
import os

def create_app():
    app = Flask(__name__)

    IMG_SHAPE  = 224 # Our training data consists of images with width of 224 pixels and height of 224 pixels
    MODEL_DIR = "/mnt/c/Users/bruno/Documents/1_Programming/z-temp/Models"
    MODEL_NAME = "model_CatDog.keras"

    modelPath = os.path.join(MODEL_DIR, MODEL_NAME)

    # Logging setup
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(os.path.join(log_dir, 'app.log'), maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.DEBUG)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG)

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

    return app
