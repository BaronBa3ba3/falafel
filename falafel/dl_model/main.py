import os
import sys
sys.path.append(os.path.abspath('falafel/dl_model'))

import constants as constants
import getDatabase as getDatabase
import createModel as createModel
import trainModel as trainModel
import predict as predict

# import falafel.dl_model.constants as constants
# import falafel.dl_model.getDatabase as getDatabase
# import falafel.dl_model.createModel as createModel
# import falafel.dl_model.trainModel as trainModel
# import falafel.dl_model.predict as predict



def main():

#### Defining Variables

    # os.chdir(r'falafel/dl_model_developpement')     # Change working directory. Or could have contacate all directories in constants.py with the relative dir : MODEL_DEVELOPPEMENT_DIR + MODEL_DIR


#### Calling Functions


    print("\n\t[1/5] - Getting Database\n")
    getDatabase.main()



    if os.path.isfile(constants.MODEL_PATH):
        print("\n\t[2/5] - Model already Created\n")
    else:
        print("\n\t[2/5] - Creating Model\n")
        createModel.main()


    if (constants.TRAIN_BOOL == 0):
        print("\n\t[3/5] - Model already trained\n")
    else:
        print("\n\t[3/5] - Training Model\n")
        trainModel.main()



    print("\n\t[4/5] - Predicting Images\n")
    predict.main()



    print("\n\t[5/5] - Finished : Model Creation completed\n")





if __name__ == "__main__":
    main()
