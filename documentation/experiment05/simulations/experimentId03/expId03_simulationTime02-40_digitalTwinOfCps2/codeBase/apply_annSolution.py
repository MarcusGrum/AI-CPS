""" 
This application categorizes transport situations on behalf of pretrained artificial neural networks (ANN).
For this, it loads an ANN-based solution (pickle-file / xml-file) from the 'knowledgeBase' of docker volume 'ai_system' 
and activates it with image classification outcomes from the 'activationBase' of docker volume 'ai_system'.
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
import sys, os
import timeit
##sys.path.append('localPathToCorrespondingPyBrainLibrary')

from pybrain import *
#from pybrain.utilities           import percentError
#from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
#from pybrain.structure.modules   import SoftmaxLayer, SigmoidLayer, TanhLayer, LSTMLayer
from pybrain.datasets import SupervisedDataSet, SequentialDataSet, ClassificationDataSet
import numpy as np
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import axes3d

import pickle
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
import csv

# constant definition
net = None

def load_network_from_file_by_pickle(path, name):
    """
    This functions loads a pickle network file and returns the network by the element called 'net'.
    """ 
    fileObject = open(path + name + '.pickle', 'rb')
    net = pickle.load(fileObject)
    
    return net

def load_network_from_file_by_NetworkReader(path, name):
    """
    This functions loads a xml network file and returns the network by the element called 'net'.
    """ 
    net = NetworkReader.readFrom(path + name + '.xml') 
    
    return net

def codify_currentClassificationOutput(activationInput, currentImmageClassificationOutput_typeOfFruit, currentImmageClassificationOutput_quality):
    """
    This function codifies given activation output of previously realized image classification, 
    then integrates codified activation with the current activation input of activationInput 
    and returns it.
    """

    if (currentImmageClassificationOutput_typeOfFruit == 'apple'):
        if (currentImmageClassificationOutput_quality == 'def'):
            activationInput[0] = 1
        elif (currentImmageClassificationOutput_quality == 'ok'):
            activationInput[1] = 1
    elif (currentImmageClassificationOutput_typeOfFruit == 'banana'):
        if (currentImmageClassificationOutput_quality == 'def'):
            activationInput[2] = 1
        elif (currentImmageClassificationOutput_quality == 'ok'):
            activationInput[3] = 1
    elif (currentImmageClassificationOutput_typeOfFruit == 'orange'):
        if (currentImmageClassificationOutput_quality == 'def'):
            activationInput[4] = 1
        elif (currentImmageClassificationOutput_quality == 'ok'):
            activationInput[5] = 1
    elif (currentImmageClassificationOutput_typeOfFruit == 'pump'):
        if (currentImmageClassificationOutput_quality == 'def'):
            activationInput[6] = 1
        elif (currentImmageClassificationOutput_quality == 'ok'):
            activationInput[7] = 1

    return activationInput

def decodify_currentClassificationOutput (result_rounded):
    """
    This function decodifies generated activation output of current transport classification, 
    and returns its interpretation.
    """

    interpretation = ''
    if ((result_rounded[0] == 1) and (result_rounded[1] == 0) and (result_rounded[2] == 0)):
        interpretation = 'transport-to-the-right'
    if ((result_rounded[0] == 0) and (result_rounded[1] == 1) and (result_rounded[2] == 0)):
        interpretation = 'idling'
    if ((result_rounded[0] == 0) and (result_rounded[1] == 0) and (result_rounded[2] == 1)):
        interpretation = 'transport-to-the-left'
    if (result_rounded[3] == 1):
        interpretation = interpretation + '-alarm'
    if (result_rounded[4] == 1):
        interpretation = interpretation + '-claim'

    return interpretation

def initializeSensorFiles(pathActivationBase):
    """
    This function initializes relevant sensor files at current 'activationBase' of docker volume 'ai_system'
    for setting up cps for simulation.
    It can be used for manually specifying sensor values for testing, too.
    """
    
    writepath = pathActivationBase+'currentActivation/'+'cps1_conveyor_workpieceSensorLeft.txt'
    if not os.path.exists(writepath):
        with open(writepath, 'w') as f:
            f.write('0.0')
    
    writepath = pathActivationBase+'currentActivation/'+'cps1_conveyor_workpieceSensorCenter.txt'
    if not os.path.exists(writepath):
        with open(writepath, 'w') as f:
            f.write('1.0')

    writepath = pathActivationBase+'currentActivation/'+'cps1_conveyor_workpieceSensorRight.txt'
    if not os.path.exists(writepath):
        with open(writepath, 'w') as f:
            f.write('0.0')

    writepath = pathActivationBase+'currentActivation/'+'cps2_conveyor_workpieceSensorLeft.txt'
    if not os.path.exists(writepath):
        with open(writepath, 'w') as f:
            f.write('0.0')

    writepath = pathActivationBase+'currentActivation/'+'cps2_conveyor_workpieceSensorCenter.txt'
    if not os.path.exists(writepath):
        with open(writepath, 'w') as f:
            f.write('0.0')

    writepath = pathActivationBase+'currentActivation/'+'cps2_conveyor_workpieceSensorRight.txt'
    if not os.path.exists(writepath):
        with open(writepath, 'w') as f:
            f.write('0.0')

def loadCurrentSensorValues_and_integrateValuesWithCurrentActivationInput(pathActivationBase, activationInput):
    """
    This function loads current sensor values from sensor files at current 'activationBase' of docker volume 'ai_system'
    and integrates them with the current activation input (in dependence of current kind of CPS1 or CPS2).
    It returns integrated activationInput.
    """

    with open(pathActivationBase+'currentActivation/'+'cps1_conveyor_workpieceSensorCenter.txt', 'r') as f:
        for line in f:
            currentValue_cps1_conveyor_workpieceSensorCenter = line
    with open(pathActivationBase+'currentActivation/'+'cps1_conveyor_workpieceSensorLeft.txt', 'r') as f:
        for line in f:
            currentValue_cps1_conveyor_workpieceSensorLeft = line
    with open(pathActivationBase+'currentActivation/'+'cps1_conveyor_workpieceSensorRight.txt', 'r') as f:
        for line in f:
            currentValue_cps1_conveyor_workpieceSensorRight = line
    with open(pathActivationBase+'currentActivation/'+'cps2_conveyor_workpieceSensorCenter.txt', 'r') as f:
        for line in f:
            currentValue_cps2_conveyor_workpieceSensorCenter = line
    with open(pathActivationBase+'currentActivation/'+'cps2_conveyor_workpieceSensorLeft.txt', 'r') as f:
        for line in f:
            currentValue_cps2_conveyor_workpieceSensorLeft = line
    with open(pathActivationBase+'currentActivation/'+'cps2_conveyor_workpieceSensorRight.txt', 'r') as f:
        for line in f:
            currentValue_cps2_conveyor_workpieceSensorRight = line

    if('cps1' in receiver):
        activationInput[ 8] = currentValue_cps1_conveyor_workpieceSensorLeft
        activationInput[ 9] = currentValue_cps1_conveyor_workpieceSensorCenter
        activationInput[10] = currentValue_cps1_conveyor_workpieceSensorRight
        activationInput[11] = currentValue_cps2_conveyor_workpieceSensorLeft
    elif('cps2' in receiver):
        activationInput[ 8] = currentValue_cps2_conveyor_workpieceSensorLeft
        activationInput[ 9] = currentValue_cps2_conveyor_workpieceSensorCenter
        activationInput[10] = currentValue_cps2_conveyor_workpieceSensorRight
        activationInput[11] = currentValue_cps1_conveyor_workpieceSensorRight

    return activationInput

def openAnnSolution():
    """
    This function opens a pretrained ANN from the 'knowledgeBase' of docker volume 'ai_system'.
    """
        
    # use model that already has been declared initially
    global net

    # specify standard path for loading ANN-based solution
    pathKnowledgeBase = "/tmp/"+sender+"/knowledgeBase/"
    
    # load solution from standard path
    net = load_network_from_file_by_pickle(path = pathKnowledgeBase, name = 'currentSolution')
    net = load_network_from_file_by_NetworkReader(path = pathKnowledgeBase, name = 'currentSolution')
    
    # indicate successful loading by CLI output
    print("...solution has been loaded successfully!")
    
    return net

def applyAnnSolution():
    """
    This function applies the ANN loaded to images from the 'activationBase' of docker volume 'ai_system'.
    """

    # specify standard path for loading images from activationBase
    pathActivationBase = "/tmp/"+sender+"/activationBase/"

    # initialize cps for simulation by creating sensor stream files
    activationInput = np.zeros(12)
    # for manual debugging only
    #initializeSensorFiles(pathActivationBase)

    # start time measurement (for evaluation purposes)
    start = timeit.default_timer()

    # acquire image data classification output from activationBase
    with open(pathActivationBase+'currentApplicationResults.txt', 'r') as f:
        for line in f:
            currentImmageClassificationOutput = line.strip().split("-> ",1)[1]
            currentImmageClassificationOutput_typeOfFruit = currentImmageClassificationOutput.split("-",1)[0]
            currentImmageClassificationOutput_quality = currentImmageClassificationOutput.split("-",1)[1]
            print('fruit=' + currentImmageClassificationOutput_typeOfFruit + ', quality=' + currentImmageClassificationOutput_quality)
            break # as we are just interested in the first line
    
    # codify current image classification output
    activationInput = codify_currentClassificationOutput(activationInput, currentImmageClassificationOutput_typeOfFruit, currentImmageClassificationOutput_quality)

    # acquire further sensor data
    activationInput = loadCurrentSensorValues_and_integrateValuesWithCurrentActivationInput(pathActivationBase, activationInput)
    print('activationInput = ', activationInput)
    
    # calculate predictions for input activation
    net.reset()
    result = net.activate(activationInput)
    print('result = ', result)
    result_rounded = result.round(decimals=0)
    print('result_rounded = ', result_rounded)

    # end time measurement (for evaluation purposes)
    end = timeit.default_timer()

    # store application results at standard path of activationBase
    with open(pathActivationBase+'currentApplicationResults.txt', 'a') as f:
        f.write(str(activationInput))
        f.write(str(result_rounded))
        f.write(' -> ')
        interpretation = decodify_currentClassificationOutput (result_rounded)
        print('interpretation = ', interpretation)
        f.write(interpretation)    
        f.write('\n')

    # indicate successful ANN application by CLI output
    print("...solution has been applied successfully!")
    print("...time elapsed: " + str(round(end-start,2)))
    
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