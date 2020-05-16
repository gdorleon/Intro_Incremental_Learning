# Incremental Learning with Creme for Intel Image Classification Image Scene Classification of Multiclass


>Incremental learning is a machine learning paradigm where the learning process takes place whenever new example(s) emerge and adjusts what has been learned according to the new example(s). The most prominent difference of incremental learning from traditional machine learning is that it does not assume the availability of a sufficient training set before the learning process, but the training examples appear over time. 
_By Geng X., Smith-Miles K. (2009) Incremental Learning. In: Li S.Z., Jain A. (eds) Encyclopedia of Biometrics. Springer, Boston, MA_

## Brief Exlpanation of the Model Development Process

By applying transfer learning and feature extraction with the weights of a ```ResNet50``` Neural Network trained with ImageNet, a new set of data is created,
this new set of data is made by the extracted features from the ```ResNet50``` Neural Network, then a binary incremental learning model is applied, ```PAClassifier```, 
which is implemented with the strategy ```OneVsRestClassifier``` that consists in fitting one classifier per class, for each classifier, the class is fitted against all the other classes.

## Dataset

The dataset is [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification), and it consists of image data of Natural Scenes around the world.
The datase contains around 25k images of size 150x150 distributed under 6 categories:
* Buildings: _This class in the dataset has an integer label of 0_
* Forest: _This class in the dataset has an integer label of 1_
* Glacier: _This class in the dataset has an integer label of 2_
* Mountain: _This class in the dataset has an integer label of 3_
* Sea: _This class in the dataset has an integer label of 4_
* Street: _This class in the dataset has an integer label of 5_

### Dataset Acknowledgement
Thanks to [Datahack Analytics Vidhya](https://datahack.analyticsvidhya.com) for the challenge and Intel for the Data and the Kaggle user
[Puneet Bansal](https://www.kaggle.com/puneet6060)

## Repository Structure
* ```config.py```: This file holds configurations of files names, paths for directories of training, validation and testing sets, and the batch size for fetching images.
* ```build_dataset.py```: The dataset is built thanks to this script, it inherits file names, directories and the batch size from the ```config.py``` file. It also makes use of the ```extract_features.py``` script to execute the creation of the features set.
* ```extract_features.py```: In this file, a ```ResNet50``` Neural Network model is loaded to use as feature extractor, it loads the dataset, creates a features csv file corresponding to each set, extract the features from the batch of images and saves the features in the created file.
* ```train_incremental.py```: Online training is donde thanks to this file. During the phase of online trainig  a classification report is printed for displaying the results obtained during training, once this phase has finished the metrics from the evaluation are displayed and the model is saved into ```.pkl``` file.

## Usage
1. Download the dataset from [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification), unzip it in the same directory where these scripts are.
2. Run ```python build_dataset.py -d [directory]```, as explained above, ```build_dataset.py``` will try to create the necessary dataset features for training, validation and testing.
3. Run ```python train_incremental.py -c [path to features_set.csv] -n [100352 as value the vector space of a ResNet50 NN]```, once the command is executed the online training process will start.

### Note:
This is a work in progress, the deployment of the model will be done with [Chantilly](https://github.com/creme-ml/chantilly).
