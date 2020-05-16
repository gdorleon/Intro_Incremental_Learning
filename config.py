# import the necessary packages
import os

# initialize the path to the *original* input directory of images
ORIG_INPUT_DATASET = "intel-image-classification"

# initialize the path to the extracted features input
FEATURES_DATASET = "dataset-features"

# define names of extracted features csv files
TRAIN_FEATURES_CSV = "train_features.csv"
VAL_FEATURES_CSV = "val_features.csv"
PRED_FEATURES_CSV = "pred_features.csv"

# define the names of the training, testing, and validation directories
TRAIN_SET_PATH = os.path.join(ORIG_INPUT_DATASET,"seg_pred", "seg_pred")
VAL_SET_PATH= os.path.join(ORIG_INPUT_DATASET,"seg_test", "seg_test")
PRED_SET_PATH= os.path.join(ORIG_INPUT_DATASET,"seg_pred", "seg_pred")

# define array of sets path
PATH_SETS = [TRAIN_SET_PATH, VAL_SET_PATH, PRED_SET_PATH]

# set the batch size
BATCH_SIZE = 32

# initialize the label encoder file path and the output directory to
# where the extracted features (in CSV file format) will be stored
LE_PATH = os.path.sep.join([FEATURES_DATASET, "label", "le.cpickle"])

TRAIN_FEATURES_CSV_PATH = os.path.join(FEATURES_DATASET, "train_features", TRAIN_FEATURES_CSV)
VAL_FEATURES_CSV_PATH = os.path.join(FEATURES_DATASET, "val_features", VAL_FEATURES_CSV)
PRED_FEATURES_CSV_PATH = os.path.join(FEATURES_DATASET, "pred_features", PRED_FEATURES_CSV)

# define array of extracted features path
PATH_FEATURES_CSV_SETS = [TRAIN_FEATURES_CSV_PATH, VAL_FEATURES_CSV_PATH, PRED_FEATURES_CSV_PATH]

# Print Path to images sets
def print_images_sets_paths():
    print("TRAIN_SET PATH: {}\nVAL_SET_PATH: {}\nPRED_SET_PATH: {}\n".format(TRAIN_SET_PATH, VAL_SET_PATH, PRED_SET_PATH))

# Print Path to extracted features sets
def print_features_sets_paths():
    print("New paths has been created:\nLE_PATH: {}\nTRAIN_FEATURES_CSV_PATH: {}\nVAL_FEATURES_CSV_PATH: {}\nPRED_FEATURES_CSV_PATH: {}".format(LE_PATH, TRAIN_FEATURES_CSV_PATH, VAL_FEATURES_CSV_PATH, PRED_FEATURES_CSV_PATH))
