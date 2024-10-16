import os
import configparser
import zipfile


# Fonction to determine classes
def get_subfolder_names(directory):
    subfolder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subfolder_names.append(item)
    subfolder_names.sort()
    return subfolder_names

# Fonction to determine number of classes
def get_subfolder_number(directory):
    items = os.listdir(directory)
    
    # Filter out only the subdirectories. Ignores files
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory, item))]
    
    return len(subdirectories)

# Fonction that extracts the first 2 layers of a .zip file (main dir + first layer. ex : flowers_5/daisy)
def extract_first_layer(zip_path, extract_to):

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # List all files and directories in the zip file
        file_list = zip_ref.namelist()

        # Extract top-level and second-level items (i.e., up to two directory layers)
        for item in file_list:
            # Split the path into its components
            path_parts = item.rstrip('/').split('/')

            # Check if the item is in the first or second layer
            if len(path_parts) <= 2:
                # Create directories if it's a folder
                if item.endswith('/'):
                    os.makedirs(os.path.join(extract_to, item), exist_ok=True)
                # Extract files
                # else:
                #     zip_ref.extract(item, extract_to)



#### Importing .conf file

# Create a ConfigParser object
config = configparser.ConfigParser()

# Read the configuration file
config_file_path = 'config/falafel.conf'

config.read(config_file_path)

if config_file_path in config.read(config_file_path):
    print("SUCCESS : Configuration file '{}' was successfully read.".format(config_file_path))
else:
    if os.path.exists(config_file_path):
        print("ERROR : Configuration file '{}' exists but could not be read. Check permissions.".format(config_file_path))
    else:
        print("ERROR : Configuration file '{}' does not exist.".format(config_file_path))




#### Linking Variables

## Model Parameters

TRAIN_BOOL = config.getboolean('Model Parameters', 'TRAIN_MODEL_BOOL')              # '1' if you want to train the model, '0' otherwise
RST_MODEL_BOOL = config.getboolean('Model Parameters', 'RST_MODEL_BOOL')            # '1' if you want to reset the model, '0' otherwise
MAX_TRAIN_RUN = config.getint('Model Parameters', 'MAX_TRAIN_RUN')                  # Maximum number of runs the model will go through. '0' for unlimited.

EPOCHS = config.getint('Model Parameters', 'EPOCHS')
BATCH_SIZE = config.getint('Model Parameters', 'BATCH_SIZE')                        # Number of training examples to process before updating our models variables
IMG_SHAPE  = config.getint('Model Parameters', 'IMG_SHAPE')                         # Our training data consists of images with width of 224 pixels and height of 224 pixels
N_STEPS_PER_EPOCH = config.getint('Model Parameters', 'N_STEPS_PER_EPOCH')          # Number of steps per epoch. '0' for default value (train_length // batch_size)
VAL_SPLIT = config.getfloat('Model Parameters', 'VAL_SPLIT')                        # Ratio of images to be reserved for validation (0.15 = 15%)

MODEL_NAME = config['Model Parameters']['MODEL_NAME']                               # Defines the name of the model

MAX_CONTENT_LENGTH = config.getint('WEBSITE', 'MAX_CONTENT_LENGTH') * 1024 * 1024  # Turning number into MB


## Paths

OS_SYSTEM = config['System']['OS']

if OS_SYSTEM in ('Linux', 'Windows', 'WSL', 'Docker'):
    DATABASE_DIR = config[OS_SYSTEM]['DATABASE_DIR']
    MODEL_DIR = config[OS_SYSTEM]['MODEL_DIR']
    LOG_DIR = config[OS_SYSTEM]['LOG_DIR']
    UPLOAD_FOLDER = config[OS_SYSTEM]['UPLOAD_FOLDER']                               # Defines the name of the folder where uploaded images (from website) are stored
else:
    print("ERROR : System not supported")


DL_MODEL_SUBDIR = config['Database']['DL_MODEL_SUBDIR']                             # Defines the location of the model developpement folder (relative to the working dir)
PREDICTION_DIR_NAME = "prediction_data"                                             # Defines name of the prediction folder (images to be predicted)
PREDICTION_DIR = os.path.join(DL_MODEL_SUBDIR, PREDICTION_DIR_NAME)                 # Defines the location of the prediction folder (images to be predicted)


MODEL_HISTORY_DIR = os.path.join(MODEL_DIR, 'model_history')                        # Defines the location of the model history folder
# MODEL_HISTORY_PATH = os.path.join(MODEL_HISTORY_DIR, 'history.pkl')                 # Defines the location of the model history file

## DATABASE


DATABASE_MAME = config['Database']['DATABASE_MAME']
DATABASE_ZIP = os.path.join(DL_MODEL_SUBDIR, "data", "".join([DATABASE_MAME, ".zip"]))     # Defines location of import of database zip file


BASE_DATA_DIR = os.path.join(DATABASE_DIR, DATABASE_MAME)                           # Defines the location of the base folder for the working database
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')                                    # Defines the location of the training folder (training data)
VALIDATION_DIR = os.path.join(BASE_DATA_DIR, 'validation')                          # Defines the location of the validation folder (validation data)

GET_DATABASE = config.getboolean('Database', 'GET_DATABASE')

extract_first_layer(DATABASE_ZIP, DATABASE_DIR)
N_CLASSES = get_subfolder_number(BASE_DATA_DIR) if config.getint('Database', 'N_CLASSES') == 0 else config.getint('Database', 'N_CLASSES')   # Number of classes (plants in Database)        
CLASS_LABELS = get_subfolder_names(BASE_DATA_DIR)                                   # List of classes (labels)

## MODEL

MODEL_PATH = os.path.join(MODEL_DIR, "".join([MODEL_NAME, ".keras"]))               # Defines location of the model file



#### Creating directories

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, 'plots'), exist_ok=True)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_HISTORY_DIR, exist_ok=True)


if OS_SYSTEM == "Docker":
    os.makedirs(DATABASE_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)







