import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit

from keras.models import load_model
import keras.utils as image
from tensorflow.keras.preprocessing.image import ImageDataGenerator


from logging.handlers import RotatingFileHandler
from PIL import Image
from werkzeug.utils import secure_filename
import io
import json
import logging
import numpy as np
import time
import threading
import os

import falafel.dl_model.constants as constants
import falafel.dl_model.main as dl_model






app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')



#### Variables Declaration
IMG_SHAPE  = constants.IMG_SHAPE

modelPath = constants.MODEL_PATH

UPLOAD_FOLDER = constants.UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}
MAX_CONTENT_LENGTH = constants.MAX_CONTENT_LENGTH

# Global variables
training_thread = None
is_training = False
current_epoch = 0
total_epochs = 0

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


def train_model(epochs):
    global model, is_training, current_epoch, total_epochs
    is_training = True
    total_epochs = epochs
    current_epoch = 0

    ## My variables
    train_dir = constants.TRAIN_DIR
    validation_dir = constants.VALIDATION_DIR
    EPOCHS = constants.EPOCHS
    BATCH_SIZE = constants.BATCH_SIZE  # Number of training examples to process before updating our models variables
    IMG_SHAPE  = constants.IMG_SHAPE # Our training data consists of images with width of 224 pixels and height of 224 pixels
    total_train = sum([len(files) for r, d, files in os.walk(train_dir)]) # Number of Training Images
        # Number of Steps per Epoch
    if (constants.N_STEPS_PER_EPOCH == 0):
        n_steps_epoch = int(np.ceil(total_train / float(BATCH_SIZE)))
    else:
        n_steps_epoch = constants.N_STEPS_PER_EPOCH


    train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
    train_generator = train_datagen.flow_from_directory(train_dir, batch_size = BATCH_SIZE, class_mode = 'binary', target_size = (IMG_SHAPE, IMG_SHAPE), subset='training')

    test_datagen = ImageDataGenerator( rescale = 1.0/255. )
    validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = BATCH_SIZE, class_mode = 'binary', target_size = (IMG_SHAPE, IMG_SHAPE), subset='validation')
    

    try:
        for epoch in range(epochs):
            if not is_training:
                break
            current_epoch = epoch + 1
            history = model.fit(
                train_generator,
                steps_per_epoch=train_generator.samples // train_generator.batch_size,
                validation_data=validation_generator,
                validation_steps=validation_generator.samples // validation_generator.batch_size,
                epochs=1,
                verbose=0
            )
            
            # Emit progress
            progress = {
                'epoch': current_epoch,
                'total_epochs': total_epochs,
                'loss': history.history['loss'][0],
                'accuracy': history.history['accuracy'][0],
                'val_loss': history.history['val_loss'][0],
                'val_accuracy': history.history['val_accuracy'][0]
            }
            socketio.emit('training_progress', progress)
            
            # Save model after each epoch
            model.save(modelPath)
            
            with open('training_state.json', 'w') as f:
                json.dump({'current_epoch': current_epoch, 'total_epochs': total_epochs}, f)
    
    except Exception as e:
        app.logger.error(f"Error during training: {str(e)}")
        socketio.emit('training_error', {'error': str(e)})
    finally:
        is_training = False
        socketio.emit('training_complete')





    



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
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            app.logger.info(f"File saved to {filepath}")
            results = analyze_image(filepath)
            return jsonify({'results': results})
        return jsonify({'error': 'File type not allowed'}), 400
    except Exception as e:
        app.logger.error(f"Error in upload_file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_thread, is_training
    if is_training:
        return jsonify({'error': 'Training is already in progress'}), 400
    
    epochs = request.json.get('epochs', 10)  # Default to 10 epochs if not specified
    training_thread = threading.Thread(target=train_model, args=(epochs,))
    training_thread.start()
    return jsonify({'message': 'Training started'})

@app.route('/stop_training', methods=['POST'])
def stop_training():
    global is_training
    is_training = False
    return jsonify({'message': 'Training stopped'})

@socketio.on('connect')
def handle_connect():
    global current_epoch, total_epochs
    if os.path.exists('training_state.json'):
        with open('training_state.json', 'r') as f:
            state = json.load(f)
            current_epoch = state['current_epoch']
            total_epochs = state['total_epochs']
    emit('training_state', {'is_training': is_training, 'current_epoch': current_epoch, 'total_epochs': total_epochs})


@socketio.on('connect')
def handle_connect():
    global current_epoch, total_epochs
    if os.path.exists('training_state.json'):
        with open('training_state.json', 'r') as f:
            state = json.load(f)
            current_epoch = state['current_epoch']
            total_epochs = state['total_epochs']
    emit('training_state', {'is_training': is_training, 'current_epoch': current_epoch, 'total_epochs': total_epochs})

