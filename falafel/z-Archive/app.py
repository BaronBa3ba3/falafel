import os
import io

from flask import Flask, request, jsonify
import numpy as np

from tensorflow.python.keras.models import load_model
from keras.utils import img_to_array

from PIL import Image


#### Variables 

IMG_SHAPE  = 224 # Our training data consists of images with width of 224 pixels and height of 224 pixels
MODEL_DIR = "/mnt/c/Users/bruno/Documents/1_Programming/z-temp/Models"
MODEL_NAME = "model_CatDog.keras"

modelPath = os.path.join(MODEL_DIR, MODEL_NAME)

print(os.path.abspath(modelPath))

app = Flask(__name__)
model = load_model(modelPath)

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)       # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if request.method == "POST":
        if request.files.get("image"):
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, target=(IMG_SHAPE, IMG_SHAPE))  # Use your model's input size

            preds = model.predict(image)
            data["predictions"] = preds.tolist()
            data["success"] = True

    return jsonify(data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)



## Request :
#
# curl -X POST -F "image=@data_test/pig.jpg" http://localhost:5000/predict
