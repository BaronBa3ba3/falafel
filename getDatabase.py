import os
import zipfile 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def main():


	#### Preparing the dataset

	local_zip = 'data/cats_and_dogs_filtered.zip'
	zip_ref = zipfile.ZipFile(local_zip, 'r')
	#zip_ref.extractall('/tmp')
	zip_ref.extractall("C:/Users/bruno/Documents/1_Programming/z-temp")
	zip_ref.close()

	#base_dir = '/tmp/cats_and_dogs_filtered'
	base_dir = 'C:/Users/bruno/Documents/1_Programming/z-temp/cats_and_dogs_filtered'
	train_dir = os.path.join(base_dir, 'train')
	validation_dir = os.path.join(base_dir, 'validation')

	# Directory with our training cat pictures
	train_cats_dir = os.path.join(train_dir, 'cats')

	# Directory with our training dog pictures
	train_dogs_dir = os.path.join(train_dir, 'dogs')

	# Directory with our validation cat pictures
	validation_cats_dir = os.path.join(validation_dir, 'cats')

	# Directory with our validation dog pictures
	validation_dogs_dir = os.path.join(validation_dir, 'dogs')



#### Visualizing some of the training images

	nrows = 4
	ncols = 4

	fig = plt.gcf()
	fig.set_size_inches(ncols*4, nrows*4)
	pic_index = 100
	train_cat_fnames = os.listdir( train_cats_dir )
	train_dog_fnames = os.listdir( train_dogs_dir )


	next_cat_pix = [os.path.join(train_cats_dir, fname) 
					for fname in train_cat_fnames[ pic_index-8:pic_index] 
				]

	next_dog_pix = [os.path.join(train_dogs_dir, fname) 
					for fname in train_dog_fnames[ pic_index-8:pic_index]
				]

	for i, img_path in enumerate(next_cat_pix+next_dog_pix):
		# Set up subplot; subplot indices start at 1
		sp = plt.subplot(nrows, ncols, i + 1)
		sp.axis('Off') # Don't show axes (or gridlines)

		img = mpimg.imread(img_path)
		plt.imshow(img)

	plt.close('all') # remove to show images
	
	plt.show()


	return ([train_dir, validation_dir])




if __name__ == "__main__":
    main()