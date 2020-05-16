# USAGE
# python train_incremental.py --csv features.csv --cols 100352

# import the necessary packages
from creme.linear_model import PAClassifier
from creme.multiclass import OneVsRestClassifier
from creme.preprocessing import StandardScaler
from creme.compose import Pipeline
from creme.metrics import ClassificationReport
from creme import stream
import argparse
import numpy as np
import pickle

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--csv", required=True,
	help="path to features CSV file")
ap.add_argument("-n", "--num_cols", type=int, required=True,
	help="# of feature columns in the CSV file (excluding class column")
args = vars(ap.parse_args())

# construct our data dictionary which maps the data types of the
# columns in the CSV file to built-in data types
print("[INFO] building column names...")
types = {f'feat_{i}': float for i in range(args['num_cols'])}
types["class"] = int

# create a CSV data generator for the extracted Keras features
dataset = stream.iter_csv(filepath_or_buffer=args["csv"], target_name="class", converters=types)

# construct our pipeline
model = Pipeline(StandardScaler(), OneVsRestClassifier(binary_classifier=PAClassifier()))

# initialize our metric
print("[INFO] starting training...")
metric = ClassificationReport()

# loop over the dataset
for i, (X, y) in enumerate(dataset):
	# make predictions on the current set of features, train the
	# model on the features, and then update our metric
	preds = model.predict_one(X)
	model = model.fit_one(X, y)
	metric = metric.update(y, preds)
	print("[INFO] update {} - {}".format(i, metric))

# show the accuracy of the model
print("[INFO] final - {}".format(metric))

# save model
with open('model.pkl', 'wb') as f:
	pickle.dump(model, f)