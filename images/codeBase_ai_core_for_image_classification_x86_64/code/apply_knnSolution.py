### import libraries ###
## full imports ##
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import timeit

## selective imports ##
from os import system, name
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator


### define contstants and global values, clear console ###
## constants ##
# max of image pre-process xception3
IMG_SIZE = 299

# learning_rate
LR = 0.0001 

## global values ##
# default class-labels for example
labels =    ["apple-def","apple-ok",
            "banana-def","banana-ok",
            "orange-def","orange-ok",
            "pump-def","pump-ok"]

# init empty model
model = None

# open an existing model
def openKnnSolution():
    # use global value model (write)
    global model

    #path = input("please enter a path of where to load the model: ")
    path = "/tmp/knowledgeBase/currentSolution.h5"
    model = tf.keras.models.load_model(path)
    model.summary()
    print("success if model parameters show")
    
    return

# test single or multiple images from certain folder
def applyKnnSolution():

    #path = input("please enter a path [->folder<- with subfolder/s containing image/s]: ")
    path = "/tmp/activationBase/"

    # start take time to calculate predictions (for evaluation purposes)
    start = timeit.default_timer()

    # acquire data as during model generation
    test_batch = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=path, target_size=(IMG_SIZE,IMG_SIZE) , shuffle = False)

    # calculate predictions for input image/s
    predictions = model.predict(test_batch)

    # end take time to calculate predictions (for evaluation purposes)
    end = timeit.default_timer()

    # print filepath/s and prediction/s to terminal
    with open(path+'currentApplicationResults.txt', 'w') as f:
    	for i in range(len(predictions)):
        	mindex = np.argmax(predictions[i])
        	output_text = test_batch.filepaths[i] + " -> " + str(labels[mindex])
        	print(output_text)
        	f.write("%s\n" % output_text)

    # print time to calculate predictions (for evaluation purposes)
    print("time elapsed: " + str(round(end-start,2)) + "s for " + str(i+1) + " predictions")
    return

def main() -> int:
    """Echo the input arguments to standard output"""
    
    openKnnSolution()
    applyKnnSolution()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())