# USAGE
# python build_dataset.py

# import the necessary packages
import config
from extract_features import extract_features_from_set
import shutil
import os
import argparse
from imutils import paths

# create FEATURES_DATASET and SETs folders if not exists
if not os.path.exists(config.FEATURES_DATASET):
    # create train, val, pred folders
    os.makedirs(config.TRAIN_FEATURES_CSV_PATH)
    os.makedirs(config.VAL_FEATURES_CSV_PATH)
    os.makedirs(config.PRED_FEATURES_CSV_PATH)
    os.makedirs(config.LE_PATH)

    # print message displaying the new created folders
    config.print_features_sets_paths()


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False, default=config.ORIG_INPUT_DATASET,
	help="path to input dataset")
ap.add_argument("-b", "--batch-size", required=False, type=int, default=config.BATCH_SIZE,
	help="batch size for the network")
args = vars(ap.parse_args())

# extract features
for set_path, features_set_path in zip(config.PATH_SETS, config.PATH_FEATURES_CSV_SETS):
    extract_features_from_set(set_path, features_set_path, config.BATCH_SIZE)


