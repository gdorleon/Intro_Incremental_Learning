# USAGE
# python extract_features.py --dataset train --csv features.csv

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from imutils import paths
import numpy as np
import argparse
import pickle
import random
import os

# load the ResNet50 network and store the batch size in a convenience variable
def load_resnet50():
	print("Loading ResNet50 Neural Network...")
	model = ResNet50(weights="imagenet", include_top=False)
	return model

# grab all image paths in the input directory and randomly shuffle the paths
def get_images_from_set(dataset_path):
	imagePaths = list(paths.list_images(dataset_path))
	random.shuffle(imagePaths)
	print("The path of the images has been loaded.")
	return imagePaths

# extract the class labels from the image paths, then encode the labels
def get_labels_from_set(imagePaths):
	labels = [p.split(os.path.sep)[-2].split(".")[0] for p in imagePaths]
	le = LabelEncoder()
	labels = le.fit_transform(labels)
	print("Labels Found in Dataset: {} has been extracted and encoded.".format(le.classes_))
	return labels

# define our set of columns
def create_set_of_columns():
	cols = ["feature_{}".format(i) for i in range(0, 7 * 7 * 2048)]
	cols = ["class"] + cols
	return cols

def create_csv_feature_file(features_set_path, num_cols):
	csv = open(features_set_path, "w")
	csv.write("{}\n".format(",".join(num_cols)))
	print("A new file {} has been created, it is open." .format(features_set_path.split(os.path.sep)[-1]))
	return csv

def extract_features_from_set(set_path, features_set_path, bs):
	# get the path of the image set
	imagePaths = get_images_from_set(set_path)
	
	labels = get_labels_from_set(imagePaths)

	# load ResNet50 Neural Network
	model = load_resnet50()

	# open the CSV file for writing and write the columns names to the file
	csv = create_csv_feature_file(features_set_path, create_set_of_columns())

	# loop over the images in batches
	for (b, i) in enumerate(range(0, len(imagePaths), bs)):
		# extract the batch of images and labels, then initialize the
		# list of actual images that will be passed through the network
		# for feature extraction
		print("Processing batch {}/{}".format(b + 1, int(np.ceil(len(imagePaths) / float(bs)))))
		batchPaths = imagePaths[i:i + bs]
		batchLabels = labels[i:i + bs]
		batchImages = []

		# loop over the images and labels in the current batch
		for imagePath in batchPaths:
			# load the input image using the Keras helper utility while
			# ensuring the image is resized to 224x224 pixels
			image = load_img(imagePath, target_size=(224, 224))
			image = img_to_array(image)

			# preprocess the image by (1) expanding the dimensions and
			# (2) subtracting the mean RGB pixel intensity from the
			# ImageNet dataset
			image = np.expand_dims(image, axis=0)
			image = imagenet_utils.preprocess_input(image)

			# add the image to the batch
			batchImages.append(image)

		# pass the images through the network and use the outputs as our
		# actual features, then reshape the features into a flattened
		# volume
		batchImages = np.vstack(batchImages)
		features = model.predict(batchImages, batch_size=bs)
		features = features.reshape((features.shape[0], 7 * 7 * 2048))

		# loop over the class labels and extracted features
		for (label, vec) in zip(batchLabels, features):
			# construct a row that exists of the class label and extracted
			# features
			vec = ",".join([str(v) for v in vec])
			csv.write("{},{}\n".format(label, vec))

	# close the CSV file
	csv.close()
	print("Features extracted succesfully.")