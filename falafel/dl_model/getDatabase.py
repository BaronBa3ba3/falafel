import os
import zipfile 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import constants


def get_subfolder_names(directory):
    subfolder_names = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            subfolder_names.append(item)
    return subfolder_names



def main():

	DATABASE_DIR = constants.DATABASE_DIR
	BASE_DATA_DIR = constants.BASE_DATA_DIR

	

#### Preparing the dataset

	local_zip = constants.DATABASE_ZIP
	zip_ref = zipfile.ZipFile(local_zip, 'r')
	#zip_ref.extractall('/tmp')
	zip_ref.extractall(DATABASE_DIR)
	zip_ref.close()

	#base_dir = '/tmp/cats_and_dogs_filtered'


#### Printing numbers of images in each directory

	classes = get_subfolder_names(BASE_DATA_DIR)
	numImages = [] 									# Number of images in each class

	for flower_type in classes:
		class_directory = os.path.join(BASE_DATA_DIR, flower_type)    

		numElements = len(os.listdir(class_directory))
		numImages.append(numElements)

		print("***************************")
		print(f"Flower Type: {flower_type}")
		print(f"Number of Images: {numElements}")

	print("***************************")
	print("***************************")
	print(f"Total Number of Train Images: {int(round(sum(numImages)*(1-constants.VAL_SPLIT),0))}")
	print(f"Total Number of Valid Images: {int(round(sum(numImages)*(constants.VAL_SPLIT),0))}")
	print("***************************")







if __name__ == "__main__":
    main()