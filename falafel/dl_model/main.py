import os

import constants
import getDatabase
import createModel
import trainModel
import predict





def main():

#### Defining Variables

    # os.chdir(r'falafel/dl_model_developpement')     # Change working directory. Or could have contacate all directories in constants.py with the relative dir : MODEL_DEVELOPPEMENT_DIR + MODEL_DIR


#### Calling Functions


    print("\n\t 1-Getting Database\n")
    getDatabase.main()



    if os.path.isfile(constants.MODEL_PATH):
        print("\n\t 2-Model already Created\n")
    else:
        print("\n\t 2-Creating Model\n")
        createModel.main()


    if (constants.TRAIN_BOOL == 0):
        print("\n\t 3-Model already trained\n")
    else:
        print("\n\t 3-Training Model\n")
        trainModel.main()



    print("\n\t 4-Predicting Images\n")
    predict.main()



    print("\n\t 5-Finished\n")





if __name__ == "__main__":
    main()
