from flask import Flask, render_template, request, jsonify
from keras.models import load_model
import keras.utils as image

from PIL import Image
from logging.handlers import RotatingFileHandler
from werkzeug.utils import secure_filename
import io
import json
import logging
import numpy as np
import pickle
import plotly
import plotly.graph_objs as go
import os
import time



import falafel.dl_model.constants as constants
import falafel.dl_model.main as dl_model




def create_app():





    app = Flask(__name__)
#### Variables Declaration
    IMG_SHAPE  = constants.IMG_SHAPE
  
    modelPath = constants.MODEL_PATH

    MODEL_HISTORY_DIR = constants.MODEL_HISTORY_DIR

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
        classLabels = 'Dog' if predictions[0][0] > 0.5 else 'Cat'
        percentage = predictions[0][0] if (classLabels == 'Dog') else 1 - predictions[0][0]

        ## Convert predictions to a more user-friendly format
        ## This is a placeholder - adjust based on your model's output
        # classLabels = ["Dog", "Cat"]
        # results = [
        #     {"label": f"{classLabels[i]}", "probability": float(p)}
        #     for i, p in enumerate(predictions[0])
        # ]
        results = [
            {"label": f"{classLabels}", "probability": float(percentage)}
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


    ## Rout for model information
    @app.route('/model')
    def model_info():
        # Get model summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))
        model_summary = '\n'.join(model_summary)

        # Import Model history
        nHistories = len([name for name in os.listdir(MODEL_HISTORY_DIR) if os.path.isfile(os.path.join(MODEL_HISTORY_DIR, name))])
        history_file_names = (os.listdir(MODEL_HISTORY_DIR))
        history_array = []
        for i in range(nHistories):
            history_file = os.path.join(MODEL_HISTORY_DIR, history_file_names[i])
            with open(history_file, "rb") as file_pi:
                history_array.append(pickle.load(file_pi))


        # Create accuracy plot
        # acc_fig = go.Figure()
        # for i in range(nHistories):
        #     history_current = history_array[i]
        #     acc_fig.add_trace(go.Scatter(y=history_current['acc'], name="Train Accuracy"))
        #     acc_fig.add_trace(go.Scatter(y=history_current['val_acc'], name="Validation Accuracy"))

        # acc_fig.update_layout(title="Model Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy")


        acc_fig = go.Figure()

        epoch_offset = 0
        for i, history_i in enumerate(history_array):
            epochs = list(range(epoch_offset, epoch_offset + len(history_i['acc'])))
            acc_fig.add_trace(go.Scatter(x=epochs, y=history_i['acc'], mode='lines+markers', name=f'Run {i+1}'))
            epoch_offset += len(history_i['acc'])

        # Update layout
        acc_fig.update_layout(
            title='Training Accuracy for Multiple Runs',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            legend_title='Runs'
        )



        acc_plot = json.dumps(acc_fig, cls=plotly.utils.PlotlyJSONEncoder)

        # Create loss plot
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=history_array[0]['loss'], name="Train Loss"))
        loss_fig.add_trace(go.Scatter(y=history_array[0]['val_loss'], name="Validation Loss"))
        loss_fig.update_layout(title="Model Loss", xaxis_title="Epoch", yaxis_title="Loss")
        loss_plot = json.dumps(loss_fig, cls=plotly.utils.PlotlyJSONEncoder)

        return render_template('model.html', model_summary=model_summary, acc_plot=acc_plot, loss_plot=loss_plot)


    return app
