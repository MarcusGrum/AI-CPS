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

### define functions ###
## helper functions ##

def plotTrainingResults(epochs, accuracy, val_accuracy, loss, val_loss):
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    #plt.show()
    plt.savefig('/tmp/learningBase/TrainingPerformance.png')
    plt.savefig('/tmp/learningBase/TrainingPerformance.pdf')

## high-level functions ##
# generates model and is able to save it to disk
def createKnnSolution():
    # use global value model (write)
    global model

    # epochs are iterations of training and validation of the model
    #epochs = int(input("Please choose how many epochs should be run [int]: "))
    epochs = 1

    # acquire data as DirectoryIterator / ImageDataGenerator with Xception-PreProcessModel
    train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory = ("/tmp/learningBase/train"), target_size = (IMG_SIZE,IMG_SIZE), classes = labels)
    valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory = ("/tmp/learningBase/validation"), target_size = (IMG_SIZE,IMG_SIZE), classes = labels)

    # extract images and label of first batch
    imgs, label = next(train_batches)

    # define layers of Sequential model
    model = Sequential([
        # Sequential model scales with label count
        Conv2D(16,3,padding="same", activation="relu", input_shape=(IMG_SIZE,IMG_SIZE,3)),
        MaxPool2D(),

        Conv2D(32, 3, padding="same", activation="relu"),
        MaxPool2D(),

        Conv2D(64, 3, padding="same", activation="relu"),
        MaxPool2D(),

        Dropout(0.4),
        Flatten(),
        
        Dense(32,activation="relu"),
        Dense(max(len(labels),2), activation="softmax"),
    ])

    # show summary of model in terminal and compile it
    model.summary()
    opt = keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ["accuracy"])

    # train in epochs according to user-input, make epochs_range reusable
    runs = model.fit(x = train_batches, validation_data = valid_batches, epochs = epochs, shuffle = True)
    epochs_range = range(epochs)

    # plot the training and validation characteristics
    plotTrainingResults(range(epochs), runs.history["accuracy"], runs.history["val_accuracy"], \
                        runs.history["loss"], runs.history["val_loss"])

    # save in hierarchical data-format (HDF5, short h5)
    name = "/tmp/knowledgeBase/currentSolution.h5"
    model.save(name, save_format="h5")
    print("model saved as " + name)
    
    return

def main() -> int:
    """Echo the input arguments to standard output"""
    
    createKnnSolution()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())