import os



TRAIN_BOOL = 1   # 1 if you want to train the model, 0 otherwise

EPOCHS = 3
BATCH_SIZE = 20  # Number of training examples to process before updating our models variables
IMG_SHAPE  = 224 # Our training data consists of images with width of 224 pixels and height of 224 pixels

MODEL_DEVELOPPEMENT_DIR = "falafel/dl_model_developpement"                          # Defines the location of the model developpement folder (relative to the working dir)

DATABASE_DIR = "C:/Users/bruno/Documents/1_Programming/z-temp/Databases"            # Defines location of the databased folder (contains differents databases)
# DATABASE_DIR = "/tmp/Databases"
DATABASE_ZIP = "data/cats_and_dogs_filtered.zip"                                    # Defines location of import of database zip file
PREDICTION_DIR = "prediction_data"                                                  # Defines location of the prediction folder (images to be predicted)


BASE_DATA_DIR = os.path.join(DATABASE_DIR, 'cats_and_dogs_filtered')                # Defines the location of the base folder for the working database
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')                                    # Defines the location of the training folder (training data)
VALIDATION_DIR = os.path.join(BASE_DATA_DIR, 'validation')                          # Defines the location of the validation folder (validation data)


MODEL_NAME = "model_CatDog"                                                         # Defines the name of the model
MODEL_DIR = "C:/Users/bruno/Documents/1_Programming/z-temp/Models"                  # Defines location of the model folder
# MODEL_DIR = "/tmp/Models"
MODEL_PATH = os.path.join(MODEL_DIR, "".join([MODEL_NAME, ".keras"]))
