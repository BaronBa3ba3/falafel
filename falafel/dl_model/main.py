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

    # os.chdir(r'falafel/dl_model_developpement')               # Change working directory. Or could have contacate all directories in constants.py with the relative dir : MODEL_DEVELOPPEMENT_DIR + MODEL_DIR
    model_created = 0                                           # Boolean to check if the model already created (Do not modify).

    GET_DATABASE = constants.GET_DATABASE
    TRAIN_BOOL = constants.TRAIN_BOOL

    GET_DATABASE = 0
    TRAIN_BOOL = 1

#### Calling Functions


    if (GET_DATABASE == 0):                             # Does not unzips database if GET_DATABASE == 0
        print("\n\t[1/5] - SkippingGetting Database\n")
    else:
        print("\n\t[1/5] - Getting Database\n")
        getDatabase.main()



    if os.path.isfile(constants.MODEL_PATH):
        print("\n\t[2/5] - Model already Created\n")
        model_created = 1
    else:
        print("\n\t[2/5] - Creating Model\n")
        model_created = 0
        createModel.main()


    if (TRAIN_BOOL == 0 and model_created == 1):      # Makes sure that the model is trained if the model was just created (even though TRAIN_BOOL is 0)
        print("\n\t[3/5] - Model already trained\n")
    else:
        print("\n\t[3/5] - Training Model\n")
        trainModel.main()



    print("\n\t[4/5] - Skipping Predicting Images\n")
    # print("\n\t[4/5] - Predicting Images\n")
    # predict.main()



    print("\n\t[5/5] - Finished : Model Creation and Training completed\n")





if __name__ == "__main__":
    main()
