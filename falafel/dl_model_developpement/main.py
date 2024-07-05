import os
import subprocess

import constants
import getDatabase
import createModel
import trainModel
import predict





def main():

#### Defining Variables

    TRAIN_BOOL = constants.TRAIN_BOOL   # 1 if you want to train the model, 0 otherwise

    EPOCHS = constants.EPOCHS
    BATCH_SIZE = constants.BATCH_SIZE  # Number of training examples to process before updating our models variables
    IMG_SHAPE  = constants.IMG_SHAPE # Our training data consists of images with width of 224 pixels and height of 224 pixels

    MODEL_NAME = constants.MODEL_NAME

    DATABASE_DIR = constants.DATABASE_DIR
    MODEL_DIR = constants.MODEL_DIR

    modelPath = os.path.join(MODEL_DIR, "".join([MODEL_NAME, ".keras"]))

#### Calling Functions


    print("\n\t 1-Getting Database\n")
    directories = getDatabase.main()



    if os.path.isfile(modelPath):
        print("\n\t 2-Model already Created\n")
    else:
        print("\n\t 2-Creating Model\n")
        createModel.main(IMG_SHAPE, modelPath)


    if (TRAIN_BOOL == 0):
        print("\n\t 3-Model already trained\n")
    else:
        print("\n\t 3-Training Model\n")
        trainModel.main(directories, [EPOCHS, BATCH_SIZE, IMG_SHAPE], modelPath)



    print("\n\t 4-Predicting Images\n")
    predict.main()



    print("\n\t 5-Finished\n")





if __name__ == "__main__":
    main()
