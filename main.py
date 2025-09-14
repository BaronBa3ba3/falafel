import os
import subprocess

import getDatabase
import createModel
import trainModel
import predict





def main():

#### Defining Variables

    EPOCHS = 3
    BATCH_SIZE = 20  # Number of training examples to process before updating our models variables
    IMG_SHAPE  = 224 # Our training data consists of images with width of 224 pixels and height of 224 pixels

    MODEL_NAME = "model_CatDog"



    modelPath = "".join(["models/", MODEL_NAME, ".keras"])

#### Calling Functions

    '''
    print("\n\t 1-Getting Database\n")
    directories = getDatabase.main()



    if os.path.isfile(modelPath):
        print("\n\t 2-Model already Created\n")
    else:
        print("\n\t 2-Creating Model\n")
        createModel.main(IMG_SHAPE, modelPath)



    print("\n\t 3-Training Model\n")
    trainModel.main(directories, [EPOCHS, BATCH_SIZE, IMG_SHAPE], modelPath)

    '''

    print("\n\t 4-Predicting Images\n")
    predict.main(modelPath, IMG_SHAPE)



    print("\n\t 5-Finished\n")





if __name__ == "__main__":
    main()
