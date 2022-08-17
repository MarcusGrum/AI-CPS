""" 
This application creates and prepares artificial neural networks (ANN) for categorizing images.
For this, it loads training and test material (images) from the 'learningBase' of docker volume 'ai_system',
creates an ANN and carries out training and validation routines.
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

__author__ = 'Marcus Grum, marcus.grum@uni-potsdam.de'
# SPDX-License-Identifier: AGPL-3.0-or-later or individual license
# SPDX-FileCopyrightText: 2022 Marcus Grum <marcus.grum@uni-potsdam.de>

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

def createAnnSolution():
    """
    This function realizes the following:
    (1) It wires a new ANN on the base of an architecture specified,
    (2) it loads training material from the 'learningBase' of docker volume 'ai_system',
    (3) it carries out training and test routines,
    (4) it stores training and validation performance and
    (5) it stores the ANN trained at the 'knowledgeBase' of docker volume 'ai_system'.
    """
    
    # use model that already has been declared initially
    global model

    # epochs are iterations of training and validation of the model
    epochs = 1

    # acquire image data from learningBase
    train_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory = ("/tmp/"+sender+"/learningBase/train"), target_size = (IMG_SIZE,IMG_SIZE), classes = labels)
    valid_batches = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory = ("/tmp/"+sender+"/learningBase/validation"), target_size = (IMG_SIZE,IMG_SIZE), classes = labels)

    # extract images and labels of first batch
    imgs, label = next(train_batches)

    # specify ANN architecture
    model = Sequential([
    
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

    # show solution summary
    model.summary()
    
    # prepare training method
    opt = keras.optimizers.Adam(learning_rate=LR)
    
    # compile model by learning context specified
    model.compile(optimizer = opt , loss = 'categorical_crossentropy' , metrics = ["accuracy"])

    # train in epochs according to user-input, make epochs_range reusable
    runs = model.fit(x = train_batches, validation_data = valid_batches, epochs = epochs, shuffle = True)
    epochs_range = range(epochs)

    # plot the training and validation performance
    plotTrainingAndValidationPerformance(range(epochs), runs.history["accuracy"], runs.history["val_accuracy"], \
                        runs.history["loss"], runs.history["val_loss"])

    # specify standard path for storing ANN-based solution
    pathKnowledgeBase = "/tmp/"+sender+"/knowledgeBase/"
    
    # save solution in hierarchical data-format (HDF5, short h5)
    model.save(pathKnowledgeBase+"currentSolution.h5", save_format="h5")
    
    # indicate successful solution storing by CLI output
    print("...solution has been stored at " + pathKnowledgeBase + " successfully!")
    
    return

def main() -> int:
    """
    This function is loaded by application start first.
    It returns 0 so that corresponding docker container is stopped.
    For instance, limited raspberry resources can be unleashed and energy can be saved.
    """
    
    createAnnSolution()
    
    return 0

if __name__ == '__main__':

    # input parameters from CLI
    sender = sys.argv[1]
    receiver = sys.argv[2]
    
    # output parameters to CLI
    sys.exit(main())
