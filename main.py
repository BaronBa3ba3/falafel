import os
import tensorflow as tf 
# from tensorflow.keras.preprocessing.image import ImageDataGenerator 
# from tensorflow.keras import layers 
# from tensorflow.keras import Model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import subprocess

import getDatabase




def main():
#### Run script using subprocess

    # result = subprocess.run(['python', 'getDatabase.py'], capture_output=True, text=True)
    # # To capture output
    # print(result.stdout)
    # # To handle errors
    # if result.returncode != 0:
    #     print(f"Error: {result.stderr}")


    getDatabase.main()





if __name__ == "__main__":
    main()
