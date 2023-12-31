""" 
This application categorizes images on behalf of pretrained artificial neural networks (ANN).
For this, it loads an ANN-based solution (h5-file) from the 'knowledgeBase' of docker volume 'ai_system' 
and activates it with images from the 'activationBase' of docker volume 'ai_system'.
Then, activation results are stored at the 'activationBase' of docker volume 'ai_system'.

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

def openAnnSolution():
    """
    This function opens a pretrained ANN from the 'knowledgeBase' of docker volume 'ai_system'.
    """
        
    # use model that already has been declared initially
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

def applyAnnSolution():
    """
    This function applies the ANN loaded to images from the 'activationBase' of docker volume 'ai_system'.
    """

    # specify standard path for loading images from activationBase
    pathActivationBase = "/tmp/"+sender+"/activationBase/"

    # start time measurement (for evaluation purposes)
    start = timeit.default_timer()

    # acquire image data from activationBase
    test_batch = ImageDataGenerator(preprocessing_function = tf.keras.applications.xception.preprocess_input) \
    .flow_from_directory(directory=pathActivationBase, target_size=(IMG_SIZE,IMG_SIZE) , shuffle = False)

    # calculate predictions for input image/s
    predictions = model.predict(test_batch)

    # end time measurement (for evaluation purposes)
    end = timeit.default_timer()

    # store application results at standard path of activationBase
    with open(pathActivationBase+'currentApplicationResults.txt', 'w') as f:
        for i in range(len(predictions)):
            mindex = np.argmax(predictions[i])
            output_text = test_batch.filepaths[i] + " -> " + str(labels[mindex])
            
            # indicate ANN application result by CLI output
            print(output_text)
            
            # indicate ANN application result by file output
            f.write("%s\n" % output_text)

    # indicate successful ANN application by CLI output
    print("...solution has been applied successfully!")
    print("...time elapsed: " + str(round(end-start,2)) + "s for " + str(i+1) + " predictions")
    
    return

def main() -> int:
    """
    This function is loaded by application start first.
    It returns 0 so that corresponding docker container is stopped.
    For instance, limited raspberry resources can be unleashed and energy can be saved.
    """
    
    openAnnSolution()
    applyAnnSolution()
    
    return 0

if __name__ == '__main__':
    
    # input parameters from CLI
    sender = sys.argv[1]
    receiver = sys.argv[2]
    
    # output parameters to CLI
    sys.exit(main())
