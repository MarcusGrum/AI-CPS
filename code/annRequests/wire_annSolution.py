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

def wireAnnSolution():
    """
    This function realizes the following:
    (1) It wires a new ANN on the base of an architecture specified,
    (-) -------------------------------------------------,
    (-) -------------------------------------------------,
    (-) ---------------------------------------------- and
    (5) it stores the ANN trained at the 'knowledgeBase' of docker volume 'ai_system'.
    """
    
    # use model that already has been declared initially
    global model

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
	
	# specify standard path for storing ANN-based solution
    pathKnowledgeBase = "/tmp/"+sender+"/knowledgeBase/currentSolution.h5"
    
    # save solution in hierarchical data-format (HDF5, short h5)
    model.save(pathKnowledgeBase, save_format="h5")
    
    # indicate successful solution storing by CLI output
    print("...solution has been stored at " + pathKnowledgeBase + " successfully!")
    
    return

def main() -> int:
    """
    This function is loaded by application start first.
    It returns 0 so that corresponding docker container is stopped.
    For instance, limited raspberry resources can be unleashed and energy can be saved.
    """
    
    wireAnnSolution()
    
    return 0

if __name__ == '__main__':

    # input parameters from CLI
    sender = sys.argv[1]
    receiver = sys.argv[2]
    
    # output parameters to CLI
    sys.exit(main())
