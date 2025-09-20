from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import keras.utils as image

from PIL import Image
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import numpy as np
import io
import logging
import os
import time

import falafel.dl_model.constants as constants
import falafel.dl_model.main as dl_model




def create_app():





    app = Flask(__name__)
#### Variables Declaration
    IMG_SHAPE  = constants.IMG_SHAPE
  
    modelPath = constants.MODEL_PATH

    UPLOAD_FOLDER = constants.UPLOAD_FOLDER
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
    MAX_CONTENT_LENGTH = constants.MAX_CONTENT_LENGTH


#### Logging setup
    log_dir = constants.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)
    file_handler = RotatingFileHandler(os.path.join(log_dir, 'falafel_application.log'), maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.DEBUG)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.DEBUG)

#### Load the model
    if os.path.isfile(constants.MODEL_PATH):
        # if (constants.RST_MODEL_BOOL == 1):       # This line enters loop. need to find a way to set another variable (reset_model= 0)
        if (0 == 1):
            print('\nModel Found. Resetting Model ...\n')
            os.remove(modelPath)
            dl_model.main()

        elif(constants.TRAIN_BOOL == 1):
            print('\nModel Found. Training Model ...\n')
            dl_model.main()
        else:
            print('\nModel Found. Loading Model ...\n')

    else:
        print('\nModel not Found. Creating and Training Model ...\n')
        dl_model.main()


    model = load_model(modelPath)

    print('\nRunning WSGI Server ...\n')




#### Inner Functions

    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


    def analyze_image(filepath):
        img = image.load_img(filepath, target_size=(IMG_SHAPE, IMG_SHAPE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.

        # Simulate longer processing time
        time.sleep(3)

        predictions = model.predict(img_array)


        # Convert predictions to a more user-friendly format
        # This is a placeholder - adjust based on your model's output
        results = [
            {"label": f"Class {i}", "probability": float(p)}
            for i, p in enumerate(predictions[0])
        ]
        results.sort(key=lambda x: x['probability'], reverse=True)

        return results[:5]  # Return top 5 predictions

        



#### Routes

    ## Route for /predict. Used from terminals
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
            img = img.resize((IMG_SHAPE, IMG_SHAPE))
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
        

    ## Routs for Graphical Website
    @app.route('/', methods=['GET'])
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['POST'])
    def upload_file():
        try:
            if 'file' not in request.files:
                return jsonify({'error': 'No file part'}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(constants.UPLOAD_FOLDER, filename)
                file.save(filepath)
                app.logger.info(f"File saved to {filepath}")
                results = analyze_image(filepath)
                return jsonify({'results': results})
            return jsonify({'error': 'File type not allowed'}), 400
        except Exception as e:
            app.logger.error(f"Error in upload_file: {str(e)}")
            return jsonify({'error': str(e)}), 500



    return app
