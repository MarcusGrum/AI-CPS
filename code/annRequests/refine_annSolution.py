""" 
This application refines artificial neural networks (ANN) for categorizing images.
For this, it loads an ANN-based solution (h5-file) from the 'knowledgeBase' of docker volume 'ai_system'
as well as training and test material (images) from the 'learningBase' of docker volume 'ai_system'.
The ANN loads is refined because training and validation routines are applied.
Then, learning results (ANN in h5-file-format) are stored at the 'knowledgeBase' of docker volume 'ai_system'.

    Copyright (C) 2022>  Dr.-Ing. Marcus Grum

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
    
"""

# library imports
import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf
import timeit
from os import system, name
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator

# constant definition
IMG_SIZE = 299                        # max of image pre-process xception3
LR       = 0.0001                     # learning_rate
labels   = ["apple-def","apple-ok",
            "banana-def","banana-ok",
            "orange-def","orange-ok",
            "pump-def","pump-ok"]     # default class-labels for example
model = None

def plotTrainingAndValidationPerformance(epochs, accuracy, val_accuracy, loss, val_loss):
    """
    This function creates and stores the (1) accuracy plot and (2) loss plot
    for training and validation performances and stores them as (a) png and (b) pdf file
    at the 'learningBase' of docker volume 'ai_system'.
    - loss: categorical_crossentropy
    Computes the crossentropy loss between the labels and predictions.
    This loss is the crossentropy metric class to be used when there are multiple label classes (2 or more). 
    Here it is assumed that labels are given as a `one_hot` representation. 
    For instance, when `labels` are [2, 0, 1], `y_true` = [[0, 0, 1], [1, 0, 0], [0, 1, 0]].
    remember: a lower validation loss indicates a better model.
    - accuracy: 
    This metric creates two local variables, `total` and `count` 
    that are used to compute the frequency with which `y_pred` matches `y_true`. 
    This frequency is ultimately returned as `binary accuracy`: an idempotent operation that simply divides `total` by `count`.
    remember: a higher validation accuracy indicates a better model.
    """
    
    # initialize figure
    plt.figure(figsize=(15, 15))
    
    # characterize accuracy plot
    plt.subplot(2, 2, 1)
    plt.plot(epochs, accuracy, label="Training Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")
    
    # characterize loss plot
    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    
    # indicate performance by showing plot generated (having displays connected)
    #plt.show()
    
    # indicate performance by storing the plot as png and pdf file
    plt.savefig('/tmp/'+sender+'/learningBase/TrainingPerformance.png')
    plt.savefig('/tmp/'+sender+'/learningBase/TrainingPerformance.pdf')

def refineAnnSolution():
    """
    This function realizes the following:
    (1) it loads training material from the 'learningBase' of docker volume 'ai_system',
    (2) it carries out training and test routines of the ANN that has been loaded,
    (3) it stores training and validation performance and
    (4) it stores the ANN trained at the 'knowledgeBase' of docker volume 'ai_system'.
    """
    
    # use model that already has been loaded
    global model

    # epochs are iterations of training and validation of the model
    epochs = 1

    # acquire image data from learningBase
    train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory = ("/tmp/"+sender+"/learningBase/train"), target_size = (IMG_SIZE,IMG_SIZE), classes = labels)
    valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory = ("/tmp/"+sender+"/learningBase/validation"), target_size = (IMG_SIZE,IMG_SIZE), classes = labels)

    # extract images and label of first batch
    imgs, label = next(train_batches)

    # prepare training method
    opt = keras.optimizers.Adam(learning_rate=LR)
    
    # compile model by learning context specified
    model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ["accuracy"])

    # train in epochs according to user-input, make epochs_range reusable
    runs = model.fit(x = train_batches, validation_data = valid_batches, epochs = epochs, shuffle = True)
    epochs_range = range(epochs)

    # plot training and validation performance
    plotTrainingAndValidationPerformance(range(epochs), runs.history["accuracy"], runs.history["val_accuracy"], \
                        runs.history["loss"], runs.history["val_loss"])

    # specify standard path for storing ANN-based solution
    name = "/tmp/"+sender+"/knowledgeBase/currentSolution.h5"
    
    # save solution in hierarchical data-format (HDF5, short h5)
    model.save(name, save_format="h5")
    
    # indicate successful solution storing by CLI output
    print("...solution has been stored at " + name + " successfully!")
    
    return

def openAnnSolution():
    """
    This function opens a pretrained ANN from the 'knowledgeBase' of docker volume 'ai_system'.
    """
        
    # use global value model (write)
    global model

    # specify standard path for loading ANN-based solution
    pathKnowledgeBase = "/tmp/"+sender+"/knowledgeBase/currentSolution.h5"
    
    # load solution from standard path
    model = tf.keras.models.load_model(pathKnowledgeBase)
    
    # show solution summary
    model.summary()
    
    # indicate successful loading by CLI output
    print("...solution has been loaded successfully!")
    
    return

def main() -> int:
    """
    This function is loaded by application start first.
    It returns 0 so that corresponding docker container is stopped.
    For instance, limited raspberry resources can be unleashed and energy can be saved.
    """
    
    openAnnSolution()
    refineAnnSolution()
    
    return 0

if __name__ == '__main__':
    
    # input parameters from CLI
    sender = sys.argv[1]
    receiver = sys.argv[2]
    
    # output parameters to CLI
    sys.exit(main())
