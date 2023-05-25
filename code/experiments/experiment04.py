"""
A little script to realize an experiment,
which is either started by communication client within the CoNM environment or manually.
It realizes experiments on continual ANN training and testing on switching datasets.
For instance, it simulates the manipulation of CPS knowledge base by process change: product refinement (in context of continual learning and training data manipulation).
Copyright (c) 2023 Marcus Grum
"""

__author__ = 'Marcus Grum, marcus.grum@uni-potsdam.de'
# SPDX-License-Identifier: AGPL-3.0-or-later or individual license
# SPDX-FileCopyrightText: 2022 Marcus Grum <marcus.grum@uni-potsdam.de>

import subprocess
from unicodedata import decimal
import paho.mqtt.client as mqtt
from multiprocessing import Process, Queue, current_process, freeze_support
import time
import csv
import os
import platform
import numpy
import matplotlib.pyplot as plt

# import functions for experiment realization
import sys
sys.path.insert(0, '../messageClient')
import AI_simulation_basis_communication_client as aiClient

# specify global variables, so that they are known (1) at messageClient start and (2) at function calls from external scripts
global logDirectory
logDirectory = "./../messageClient/logs"

def load_data_fromfile(path):
    """
    This functions loads csv data from the 'path' and returns it.
    Remember, the data returned needs to be reshaped because it is flat.
    E.g. by data.reshape((maxNumberOfExperiments, maxIterationsInPhase1+maxIterationsInPhase2+1, maxMachines*maxValidationSets*maxStreams, maxNumberOfKPIs))
    """

    data = numpy.fromfile(path,sep=',',dtype=float)

    return data

def save_data_tofile(numpyArray, path):
    """
    This functions saves the data of variable 'numpyArray to the 'path'.
    """

    numpyArray.tofile(path,sep=',',format='%10.5f')

def load_data_from_CsvFile(path):
    """
    This functions loads csv data from the 'path' and returns it.
    """
    data = []
    with open(path + '.csv', newline='') as csvfile:
        # alternative delimiters '\t', ';', alternative quotechars '"', '|'
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"')
        for row in spamreader:
            data.append(row)

    return data

def save_data_to_CsvFile(listOfResults, path):
    """
    This functions saves the simulation results of variable 'listOfResults to the 'path'.
    """
    with open(path + '.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter='\t')
        for i in range(len(listOfResults)):
            wr.writerow(listOfResults[i])

def plotEvaluationPerformance(
    title_plots, iterations, 
    title_plot1, training_accuracy_plot1, testing_accuracy_plot1, training_loss_plot1, testing_loss_plot1,
    title_plot2, training_accuracy_plot2, testing_accuracy_plot2, training_loss_plot2, testing_loss_plot2,
    title_plot3, training_accuracy_plot3, testing_accuracy_plot3, training_loss_plot3, testing_loss_plot3,
    title_plot4, training_accuracy_plot4, testing_accuracy_plot4, training_loss_plot4, testing_loss_plot4,
    title_plot5, training_accuracy_plot5, testing_accuracy_plot5, training_loss_plot5, testing_loss_plot5,
    title_plot6, training_accuracy_plot6, testing_accuracy_plot6, training_loss_plot6, testing_loss_plot6,
    title_plot7, training_accuracy_plot7, testing_accuracy_plot7, training_loss_plot7, testing_loss_plot7,
    title_plot8, training_accuracy_plot8, testing_accuracy_plot8, training_loss_plot8, testing_loss_plot8,
    title_plot9, training_accuracy_plot9, testing_accuracy_plot9, training_loss_plot9, testing_loss_plot9,
        training_accuracy_plot1_standardDeviation=[], testing_accuracy_plot1_standardDeviation=[], training_loss_plot1_standardDeviation=[], testing_loss_plot1_standardDeviation=[],
        training_accuracy_plot2_standardDeviation=[], testing_accuracy_plot2_standardDeviation=[], training_loss_plot2_standardDeviation=[], testing_loss_plot2_standardDeviation=[],
        training_accuracy_plot3_standardDeviation=[], testing_accuracy_plot3_standardDeviation=[], training_loss_plot3_standardDeviation=[], testing_loss_plot3_standardDeviation=[],
        training_accuracy_plot4_standardDeviation=[], testing_accuracy_plot4_standardDeviation=[], training_loss_plot4_standardDeviation=[], testing_loss_plot4_standardDeviation=[],
        training_accuracy_plot5_standardDeviation=[], testing_accuracy_plot5_standardDeviation=[], training_loss_plot5_standardDeviation=[], testing_loss_plot5_standardDeviation=[],
        training_accuracy_plot6_standardDeviation=[], testing_accuracy_plot6_standardDeviation=[], training_loss_plot6_standardDeviation=[], testing_loss_plot6_standardDeviation=[],
        training_accuracy_plot7_standardDeviation=[], testing_accuracy_plot7_standardDeviation=[], training_loss_plot7_standardDeviation=[], testing_loss_plot7_standardDeviation=[],
        training_accuracy_plot8_standardDeviation=[], testing_accuracy_plot8_standardDeviation=[], training_loss_plot8_standardDeviation=[], testing_loss_plot8_standardDeviation=[],
        training_accuracy_plot9_standardDeviation=[], testing_accuracy_plot9_standardDeviation=[], training_loss_plot9_standardDeviation=[], testing_loss_plot9_standardDeviation=[],
    ):
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

    # characterize accuracy plot with training information
    plt.subplot(2, 2, 1)
    plt.plot(iterations, training_accuracy_plot1, label=title_plot1+" Focus")
    plt.plot(iterations, training_accuracy_plot2, label=title_plot2+" Focus")
    plt.plot(iterations, training_accuracy_plot3, label=title_plot3+" Focus")
    plt.plot(iterations, training_accuracy_plot4, label=title_plot4+" Focus")
    plt.plot(iterations, training_accuracy_plot5, label=title_plot5+" Focus")
    plt.plot(iterations, training_accuracy_plot6, label=title_plot6+" Focus")
    if title_plot7 != []: plt.plot(iterations, training_accuracy_plot7, label=title_plot7+" Focus")
    if title_plot8 != []: plt.plot(iterations, training_accuracy_plot8, label=title_plot8+" Focus")
    if title_plot9 != []: plt.plot(iterations, training_accuracy_plot9, label=title_plot9+" Focus")
    if("All" in title_plots):
        plt.errorbar(iterations, training_accuracy_plot1, training_accuracy_plot1_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_accuracy_plot2, training_accuracy_plot2_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_accuracy_plot3, training_accuracy_plot3_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_accuracy_plot4, training_accuracy_plot4_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_accuracy_plot5, training_accuracy_plot5_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_accuracy_plot6, training_accuracy_plot6_standardDeviation, linestyle='None', marker='^')
        if title_plot7 != []: plt.errorbar(iterations, training_accuracy_plot7, training_accuracy_plot7_standardDeviation, linestyle='None', marker='^')
        if title_plot8 != []: plt.errorbar(iterations, training_accuracy_plot8, training_accuracy_plot8_standardDeviation, linestyle='None', marker='^')
        if title_plot9 != []: plt.errorbar(iterations, training_accuracy_plot9, training_accuracy_plot9_standardDeviation, linestyle='None', marker='^')
    plt.axvline(x=0, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=5, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=11, ymin=0., ymax=1., linestyle='dotted')
    plt.legend(loc="center right")
    plt.title("Evaluation on Training Material as Accuracy \n at " + title_plots)
    
    # characterize loss plot with training information
    plt.subplot(2, 2, 2)
    plt.plot(iterations, training_loss_plot1, label=title_plot1+" Focus")
    plt.plot(iterations, training_loss_plot2, label=title_plot2+" Focus")
    plt.plot(iterations, training_loss_plot3, label=title_plot3+" Focus")
    plt.plot(iterations, training_loss_plot4, label=title_plot4+" Focus")
    plt.plot(iterations, training_loss_plot5, label=title_plot5+" Focus")
    plt.plot(iterations, training_loss_plot6, label=title_plot6+" Focus")
    if title_plot7 != []: plt.plot(iterations, training_loss_plot7, label=title_plot7+" Focus")
    if title_plot8 != []: plt.plot(iterations, training_loss_plot8, label=title_plot8+" Focus")
    if title_plot9 != []: plt.plot(iterations, training_loss_plot9, label=title_plot9+" Focus")
    if("All" in title_plots):
        plt.errorbar(iterations, training_loss_plot1, training_loss_plot1_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_loss_plot2, training_loss_plot2_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_loss_plot3, training_loss_plot3_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_loss_plot4, training_loss_plot4_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_loss_plot5, training_loss_plot5_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, training_loss_plot6, training_loss_plot6_standardDeviation, linestyle='None', marker='^')
        if title_plot7 != []: plt.errorbar(iterations, training_loss_plot7, training_loss_plot7_standardDeviation, linestyle='None', marker='^')
        if title_plot8 != []: plt.errorbar(iterations, training_loss_plot8, training_loss_plot8_standardDeviation, linestyle='None', marker='^')
        if title_plot9 != []: plt.errorbar(iterations, training_loss_plot9, training_loss_plot9_standardDeviation, linestyle='None', marker='^')
    plt.axvline(x=0, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=5, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=11, ymin=0., ymax=1., linestyle='dotted')
    plt.legend(loc="upper left")
    plt.title("Evaluation on Training Material as Loss \n at " + title_plots)

    # characterize accuracy plot with testing information
    plt.subplot(2, 2, 3)
    plt.plot(iterations, testing_accuracy_plot1, label=title_plot1+" Focus")
    plt.plot(iterations, testing_accuracy_plot2, label=title_plot2+" Focus")
    plt.plot(iterations, testing_accuracy_plot3, label=title_plot3+" Focus")
    plt.plot(iterations, testing_accuracy_plot4, label=title_plot4+" Focus")
    plt.plot(iterations, testing_accuracy_plot5, label=title_plot5+" Focus")
    plt.plot(iterations, testing_accuracy_plot6, label=title_plot6+" Focus")
    if title_plot7 != []: plt.plot(iterations, testing_accuracy_plot7, label=title_plot7+" Focus")
    if title_plot8 != []: plt.plot(iterations, testing_accuracy_plot8, label=title_plot8+" Focus")
    if title_plot9 != []: plt.plot(iterations, testing_accuracy_plot9, label=title_plot9+" Focus")
    if("All" in title_plots):
        plt.errorbar(iterations, testing_accuracy_plot1, testing_accuracy_plot1_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_accuracy_plot2, testing_accuracy_plot2_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_accuracy_plot3, testing_accuracy_plot3_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_accuracy_plot4, testing_accuracy_plot4_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_accuracy_plot5, testing_accuracy_plot5_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_accuracy_plot6, testing_accuracy_plot6_standardDeviation, linestyle='None', marker='^')
        if title_plot7 != []: plt.errorbar(iterations, testing_accuracy_plot7, testing_accuracy_plot7_standardDeviation, linestyle='None', marker='^')
        if title_plot8 != []: plt.errorbar(iterations, testing_accuracy_plot8, testing_accuracy_plot8_standardDeviation, linestyle='None', marker='^')
        if title_plot9 != []: plt.errorbar(iterations, testing_accuracy_plot9, testing_accuracy_plot9_standardDeviation, linestyle='None', marker='^')
    plt.axvline(x=0, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=5, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=11, ymin=0., ymax=1., linestyle='dotted')
    plt.legend(loc="center right")
    plt.title("Evaluation on Testing Material as Accuracy \n at " + title_plots)
    
    # characterize loss plot with testing information
    plt.subplot(2, 2, 4)
    plt.plot(iterations, testing_loss_plot1, label=title_plot1+" Focus")
    plt.plot(iterations, testing_loss_plot2, label=title_plot2+" Focus")
    plt.plot(iterations, testing_loss_plot3, label=title_plot3+" Focus")
    plt.plot(iterations, testing_loss_plot4, label=title_plot4+" Focus")
    plt.plot(iterations, testing_loss_plot5, label=title_plot5+" Focus")
    plt.plot(iterations, testing_loss_plot6, label=title_plot6+" Focus")
    if title_plot7 != []: plt.plot(iterations, testing_loss_plot7, label=title_plot7+" Focus")
    if title_plot8 != []: plt.plot(iterations, testing_loss_plot8, label=title_plot8+" Focus")
    if title_plot9 != []: plt.plot(iterations, testing_loss_plot9, label=title_plot9+" Focus")
    if("All" in title_plots):
        plt.errorbar(iterations, testing_loss_plot1, testing_loss_plot1_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_loss_plot2, testing_loss_plot2_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_loss_plot3, testing_loss_plot3_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_loss_plot4, testing_loss_plot4_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_loss_plot5, testing_loss_plot5_standardDeviation, linestyle='None', marker='^')
        plt.errorbar(iterations, testing_loss_plot6, testing_loss_plot6_standardDeviation, linestyle='None', marker='^')
        if title_plot7 != []: plt.errorbar(iterations, testing_loss_plot7, testing_loss_plot7_standardDeviation, linestyle='None', marker='^')
        if title_plot8 != []: plt.errorbar(iterations, testing_loss_plot8, testing_loss_plot8_standardDeviation, linestyle='None', marker='^')
        if title_plot9 != []: plt.errorbar(iterations, testing_loss_plot9, testing_loss_plot9_standardDeviation, linestyle='None', marker='^')
    plt.axvline(x=0, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=5, ymin=0., ymax=1., linestyle='dotted')
    plt.axvline(x=11, ymin=0., ymax=1., linestyle='dotted')
    plt.legend(loc="upper left")
    plt.title("Evaluation on Testing Material as Loss \n at " + title_plots)
    
    # indicate performance by showing plot generated (having displays connected)
    #plt.show()
    
    # indicate performance by storing the plot as png and pdf file
    plt.savefig(logDirectory+'/Plot_'+title_plots.replace(' ', '_')+'.png')
    plt.savefig(logDirectory+'/Plot_'+title_plots.replace(' ', '_')+'.pdf')

    # close all figures to unleash memory
    plt.close('all')

def realize_experiment_plotting(maxNumberOfExperiments = 2, maxIterationsInPhase1 = 5, maxIterationsInPhase2 = 5, maxMachines = 3, maxValidationSets = 3, maxStreams = 2, maxNumberOfKPIs = 3):
    """
    This function realizes plotting for this experiment and stores plots created at log directory.
    Here, we can find three types of plots:
    (1) individual plots per experiment run
        (AAok for run 1 / 2 / ... / maxNumberOfExperiments)
    (2) average plots over all experiment runs
        (AAok / AAdef / BBok / BBdef / OOok / OOdef)
    (3) overall plots averaging the same types of averaged experiment runs
        (Bias / Manipulation / Complement / Baseline / Baseline_ok / Baseline_def)
    """

    # 0 - accuracies, 1 - losses, 2 - n
    trainingKPIs = load_data_fromfile(path=logDirectory+'/trainingKPIs.csv').reshape((maxNumberOfExperiments, maxIterationsInPhase1+1+maxIterationsInPhase2+1, maxMachines*maxValidationSets*maxStreams, maxNumberOfKPIs))
    testingKPIs = load_data_fromfile(path=logDirectory+'/testingKPIs.csv').reshape((maxNumberOfExperiments, maxIterationsInPhase1+1+maxIterationsInPhase2+1, maxMachines*maxValidationSets*maxStreams, maxNumberOfKPIs))

    # Build plot for individual runs
    print('printing plots for each experiment run...')
    for experimentId in range(1, maxNumberOfExperiments+1, 1):
        print("  experimentId = " + str(experimentId))
        dim_1 = experimentId - 1
        for machineId in range(1, maxMachines+1, 1):
            print("    machineId = " + str(machineId))
            for streamId in range(1, maxStreams+1, 1):
                print("      streamId = " + str(streamId))
                dim_3 = ((machineId-1)*maxStreams*maxValidationSets)+((streamId-1)*maxValidationSets)
                if(streamId==1):
                    if(machineId==1):
                        prefix = "AAok"
                    elif(machineId==2):
                        prefix = "BBok"
                    elif(machineId==3):
                        prefix = "OOok"
                elif(streamId==2):
                    if(machineId==1):
                        prefix = "AAdef"
                    elif(machineId==2):
                        prefix = "BBdef"
                    elif(machineId==3):
                        prefix = "OOdef"
                plotEvaluationPerformance(
                    title_plots = "Experiment Number " + str(experimentId) + " with Focus " + prefix,
                    iterations=numpy.arange(maxIterationsInPhase1+1+maxIterationsInPhase2+1),
                    # apple-based validation
                    title_plot1 = "Apple",
                    training_accuracy_plot1=trainingKPIs[dim_1,:,dim_3+0,0],
                    testing_accuracy_plot1=testingKPIs[dim_1,:,dim_3+0,0], 
                    training_loss_plot1=trainingKPIs[dim_1,:,dim_3+0,1],
                    testing_loss_plot1=testingKPIs[dim_1,:,dim_3+0,1],
                    # banana-based validation
                    title_plot2 = "Banana",
                    training_accuracy_plot2=trainingKPIs[dim_1,:,dim_3+1,0],
                    testing_accuracy_plot2=testingKPIs[dim_1,:,dim_3+1,0], 
                    training_loss_plot2=trainingKPIs[dim_1,:,dim_3+1,1],
                    testing_loss_plot2=testingKPIs[dim_1,:,dim_3+1,1],
                    # orange-based validation
                    title_plot3 = "Orange",
                    training_accuracy_plot3=trainingKPIs[dim_1,:,dim_3+2,0],
                    testing_accuracy_plot3=testingKPIs[dim_1,:,dim_3+2,0], 
                    training_loss_plot3=trainingKPIs[dim_1,:,dim_3+2,1],
                    testing_loss_plot3=testingKPIs[dim_1,:,dim_3+2,1],
                    # okay apple-based validation
                    title_plot4 = "Okay Apple",
                    training_accuracy_plot4=trainingKPIs[dim_1,:,dim_3+3,0],
                    testing_accuracy_plot4=testingKPIs[dim_1,:,dim_3+3,0], 
                    training_loss_plot4=trainingKPIs[dim_1,:,dim_3+3,1],
                    testing_loss_plot4=testingKPIs[dim_1,:,dim_3+3,1],
                    # okay banana-based validation
                    title_plot5 = "Okay Banana",
                    training_accuracy_plot5=trainingKPIs[dim_1,:,dim_3+4,0],
                    testing_accuracy_plot5=testingKPIs[dim_1,:,dim_3+4,0], 
                    training_loss_plot5=trainingKPIs[dim_1,:,dim_3+4,1],
                    testing_loss_plot5=testingKPIs[dim_1,:,dim_3+4,1],
                    # okay orange-based validation
                    title_plot6 = "Okay Orange",
                    training_accuracy_plot6=trainingKPIs[dim_1,:,dim_3+5,0],
                    testing_accuracy_plot6=testingKPIs[dim_1,:,dim_3+5,0], 
                    training_loss_plot6=trainingKPIs[dim_1,:,dim_3+5,1],
                    testing_loss_plot6=testingKPIs[dim_1,:,dim_3+5,1],
                    # defect apple-based validation
                    title_plot7 = "Defect Apple",
                    training_accuracy_plot7=trainingKPIs[dim_1,:,dim_3+6,0],
                    testing_accuracy_plot7=testingKPIs[dim_1,:,dim_3+6,0], 
                    training_loss_plot7=trainingKPIs[dim_1,:,dim_3+6,1],
                    testing_loss_plot7=testingKPIs[dim_1,:,dim_3+6,1],
                    # defect banana-based validation
                    title_plot8 = "Defect Banana",
                    training_accuracy_plot8=trainingKPIs[dim_1,:,dim_3+7,0],
                    testing_accuracy_plot8=testingKPIs[dim_1,:,dim_3+7,0], 
                    training_loss_plot8=trainingKPIs[dim_1,:,dim_3+7,1],
                    testing_loss_plot8=testingKPIs[dim_1,:,dim_3+7,1],
                    # defect orange-based validation
                    title_plot9 = "Defect Orange",
                    training_accuracy_plot9=trainingKPIs[dim_1,:,dim_3+8,0],
                    testing_accuracy_plot9=testingKPIs[dim_1,:,dim_3+8,0], 
                    training_loss_plot9=trainingKPIs[dim_1,:,dim_3+8,1],
                    testing_loss_plot9=testingKPIs[dim_1,:,dim_3+8,1]
                    )
                dim_3 = dim_3 + maxValidationSets

    # Build plot for sum over experiments
    print('printing plots over all experiments...')
    for machineId in range(1, maxMachines+1, 1):
        print("  machineId = " + str(machineId))
        for streamId in range(1, maxStreams+1, 1):
            print("    streamId = " + str(streamId))
            dim_3 = ((machineId-1)*maxStreams*maxValidationSets)+((streamId-1)*maxValidationSets)
            if(streamId==1):
                if(machineId==1):
                    prefix = "AAok"
                elif(machineId==2):
                    prefix = "BBok"
                elif(machineId==3):
                    prefix = "OOok"
            elif(streamId==2):
                if(machineId==1):
                    prefix = "AAdef"
                elif(machineId==2):
                    prefix = "BBdef"
                elif(machineId==3):
                    prefix = "OOdef"
            plotEvaluationPerformance(
                title_plots = "Experiment Number All" + " with Focus " + prefix,
                iterations=numpy.arange(maxIterationsInPhase1+1+maxIterationsInPhase2+1),
                # apple-based validation
                title_plot1 = "Apple",
                training_accuracy_plot1=numpy.mean(trainingKPIs[:,:,dim_3+0,0], axis=0),
                testing_accuracy_plot1=numpy.mean(testingKPIs[:,:,dim_3+0,0], axis=0), 
                training_loss_plot1=numpy.mean(trainingKPIs[:,:,dim_3+0,1], axis=0),
                testing_loss_plot1=numpy.mean(testingKPIs[:,:,dim_3+0,1], axis=0),
                training_accuracy_plot1_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+0,0], axis=0),
                testing_accuracy_plot1_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+0,0], axis=0), 
                training_loss_plot1_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+0,1], axis=0),
                testing_loss_plot1_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+0,1], axis=0),
                # banana-based validation
                title_plot2 = "Banana",
                training_accuracy_plot2=numpy.mean(trainingKPIs[:,:,dim_3+1,0], axis=0),
                testing_accuracy_plot2=numpy.mean(testingKPIs[:,:,dim_3+1,0], axis=0), 
                training_loss_plot2=numpy.mean(trainingKPIs[:,:,dim_3+1,1], axis=0),
                testing_loss_plot2=numpy.mean(testingKPIs[:,:,dim_3+1,1], axis=0),
                training_accuracy_plot2_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+1,0], axis=0),
                testing_accuracy_plot2_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+1,0], axis=0), 
                training_loss_plot2_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+1,1], axis=0),
                testing_loss_plot2_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+1,1], axis=0),
                # orange-based validation
                title_plot3 = "Orange",
                training_accuracy_plot3=numpy.mean(trainingKPIs[:,:,dim_3+2,0], axis=0),
                testing_accuracy_plot3=numpy.mean(testingKPIs[:,:,dim_3+2,0], axis=0), 
                training_loss_plot3=numpy.mean(trainingKPIs[:,:,dim_3+2,1], axis=0),
                testing_loss_plot3=numpy.mean(testingKPIs[:,:,dim_3+2,1], axis=0),
                training_accuracy_plot3_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+2,0], axis=0),
                testing_accuracy_plot3_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+2,0], axis=0), 
                training_loss_plot3_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+2,1], axis=0),
                testing_loss_plot3_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+2,1], axis=0),
                # okay apple-based validation
                title_plot4 = "Okay Apple",
                training_accuracy_plot4=numpy.mean(trainingKPIs[:,:,dim_3+3,0], axis=0),
                testing_accuracy_plot4=numpy.mean(testingKPIs[:,:,dim_3+3,0], axis=0), 
                training_loss_plot4=numpy.mean(trainingKPIs[:,:,dim_3+3,1], axis=0),
                testing_loss_plot4=numpy.mean(testingKPIs[:,:,dim_3+3,1], axis=0),
                training_accuracy_plot4_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+3,0], axis=0),
                testing_accuracy_plot4_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+3,0], axis=0), 
                training_loss_plot4_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+3,1], axis=0),
                testing_loss_plot4_standardDeviation=numpy.std(testingKPIs[:,:,3,1], axis=0),
                # okay banana-based validation
                title_plot5 = "Okay Banana",
                training_accuracy_plot5=numpy.mean(trainingKPIs[:,:,dim_3+4,0], axis=0),
                testing_accuracy_plot5=numpy.mean(testingKPIs[:,:,dim_3+4,0], axis=0), 
                training_loss_plot5=numpy.mean(trainingKPIs[:,:,dim_3+4,1], axis=0),
                testing_loss_plot5=numpy.mean(testingKPIs[:,:,dim_3+4,1], axis=0),
                training_accuracy_plot5_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+4,0], axis=0),
                testing_accuracy_plot5_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+4,0], axis=0), 
                training_loss_plot5_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+4,1], axis=0),
                testing_loss_plot5_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+4,1], axis=0),
                # okay orange-based validation
                title_plot6 = "Okay Orange",
                training_accuracy_plot6=numpy.mean(trainingKPIs[:,:,dim_3+5,0], axis=0),
                testing_accuracy_plot6=numpy.mean(testingKPIs[:,:,dim_3+5,0], axis=0), 
                training_loss_plot6=numpy.mean(trainingKPIs[:,:,dim_3+5,1], axis=0),
                testing_loss_plot6=numpy.mean(testingKPIs[:,:,dim_3+5,1], axis=0),
                training_accuracy_plot6_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+5,0], axis=0),
                testing_accuracy_plot6_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+5,0], axis=0), 
                training_loss_plot6_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+5,1], axis=0),
                testing_loss_plot6_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+5,1], axis=0),
                # defect apple-based validation
                title_plot7 = "Defect Apple",
                training_accuracy_plot7=numpy.mean(trainingKPIs[:,:,dim_3+6,0], axis=0),
                testing_accuracy_plot7=numpy.mean(testingKPIs[:,:,dim_3+6,0], axis=0), 
                training_loss_plot7=numpy.mean(trainingKPIs[:,:,dim_3+6,1], axis=0),
                testing_loss_plot7=numpy.mean(testingKPIs[:,:,dim_3+6,1], axis=0),
                training_accuracy_plot7_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+6,0], axis=0),
                testing_accuracy_plot7_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+6,0], axis=0), 
                training_loss_plot7_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+6,1], axis=0),
                testing_loss_plot7_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+6,1], axis=0),
                # defect banana-based validation
                title_plot8 = "Defect Banana",
                training_accuracy_plot8=numpy.mean(trainingKPIs[:,:,dim_3+7,0], axis=0),
                testing_accuracy_plot8=numpy.mean(testingKPIs[:,:,dim_3+7,0], axis=0), 
                training_loss_plot8=numpy.mean(trainingKPIs[:,:,dim_3+7,1], axis=0),
                testing_loss_plot8=numpy.mean(testingKPIs[:,:,dim_3+7,1], axis=0),
                training_accuracy_plot8_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+7,0], axis=0),
                testing_accuracy_plot8_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+7,0], axis=0), 
                training_loss_plot8_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+7,1], axis=0),
                testing_loss_plot8_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+7,1], axis=0),
                # defect orange-based validation
                title_plot9 = "Defect Orange",
                training_accuracy_plot9=numpy.mean(trainingKPIs[:,:,dim_3+8,0], axis=0),
                testing_accuracy_plot9=numpy.mean(testingKPIs[:,:,dim_3+8,0], axis=0), 
                training_loss_plot9=numpy.mean(trainingKPIs[:,:,dim_3+8,1], axis=0),
                testing_loss_plot9=numpy.mean(testingKPIs[:,:,dim_3+8,1], axis=0),
                training_accuracy_plot9_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+8,0], axis=0),
                testing_accuracy_plot9_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+8,0], axis=0), 
                training_loss_plot9_standardDeviation=numpy.std(trainingKPIs[:,:,dim_3+8,1], axis=0),
                testing_loss_plot9_standardDeviation=numpy.std(testingKPIs[:,:,dim_3+8,1], axis=0),
                )
            dim_3 = dim_3 + maxValidationSets

    # Build plot for sum over bias / manipulation / baseline
    print('printing plots over all biases, manipulations and baselines...')
    plotEvaluationPerformance(
        title_plots = "Average Over All Experiments",
        iterations=numpy.arange(maxIterationsInPhase1+1+maxIterationsInPhase2+1),
        # bias-based validation
        title_plot1 = "Bias",
        training_accuracy_plot1 = (
            numpy.mean(trainingKPIs[:,:,0,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,9,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,19,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,28,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,38,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,47,0], axis=0)
        ) / 6,
        testing_accuracy_plot1 = (
            numpy.mean(testingKPIs[:,:,0,0], axis=0)
            + numpy.mean(testingKPIs[:,:,9,0], axis=0)
            + numpy.mean(testingKPIs[:,:,19,0], axis=0)
            + numpy.mean(testingKPIs[:,:,28,0], axis=0)
            + numpy.mean(testingKPIs[:,:,38,0], axis=0)
            + numpy.mean(testingKPIs[:,:,47,0], axis=0)
        ) / 6,
        training_loss_plot1 = (
            numpy.mean(trainingKPIs[:,:,0,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,9,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,19,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,28,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,38,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,47,1], axis=0)
        ) / 6,
        testing_loss_plot1 = (
            numpy.mean(testingKPIs[:,:,0,1], axis=0)
            + numpy.mean(testingKPIs[:,:,9,1], axis=0)
            + numpy.mean(testingKPIs[:,:,19,1], axis=0)
            + numpy.mean(testingKPIs[:,:,28,1], axis=0)
            + numpy.mean(testingKPIs[:,:,38,1], axis=0)
            + numpy.mean(testingKPIs[:,:,47,1], axis=0)
        ) / 6,
        training_accuracy_plot1_standardDeviation = (
            numpy.std(trainingKPIs[:,:,0,0], axis=0)
            + numpy.std(trainingKPIs[:,:,9,0], axis=0)
            + numpy.std(trainingKPIs[:,:,19,0], axis=0)
            + numpy.std(trainingKPIs[:,:,28,0], axis=0)
            + numpy.std(trainingKPIs[:,:,38,0], axis=0)
            + numpy.std(trainingKPIs[:,:,47,0], axis=0)
        ) / 6,
        testing_accuracy_plot1_standardDeviation = (
            numpy.std(testingKPIs[:,:,0,0], axis=0)
            + numpy.std(testingKPIs[:,:,9,0], axis=0)
            + numpy.std(testingKPIs[:,:,19,0], axis=0)
            + numpy.std(testingKPIs[:,:,28,0], axis=0)
            + numpy.std(testingKPIs[:,:,38,0], axis=0)
            + numpy.std(testingKPIs[:,:,47,0], axis=0)
        ) / 6,
        training_loss_plot1_standardDeviation = (
            numpy.std(trainingKPIs[:,:,0,1], axis=0)
            + numpy.std(trainingKPIs[:,:,9,1], axis=0)
            + numpy.std(trainingKPIs[:,:,19,1], axis=0)
            + numpy.std(trainingKPIs[:,:,28,1], axis=0)
            + numpy.std(trainingKPIs[:,:,38,1], axis=0)
            + numpy.std(trainingKPIs[:,:,47,1], axis=0)
        ) / 6,
        testing_loss_plot1_standardDeviation = (
            numpy.std(testingKPIs[:,:,0,1], axis=0)
            + numpy.std(testingKPIs[:,:,9,1], axis=0)
            + numpy.std(testingKPIs[:,:,19,1], axis=0)
            + numpy.std(testingKPIs[:,:,28,1], axis=0)
            + numpy.std(testingKPIs[:,:,38,1], axis=0)
            + numpy.std(testingKPIs[:,:,47,1], axis=0)
        ) / 6,
        # manipulation-based validation
        title_plot2 = "Manipulation",
        training_accuracy_plot2 =
            (
            numpy.mean(trainingKPIs[:,:,3,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,13,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,23,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,33,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,41,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,53,0], axis=0)
            ) / 6,
        testing_accuracy_plot2 =
            (
            numpy.mean(testingKPIs[:,:,3,0], axis=0)
            + numpy.mean(testingKPIs[:,:,13,0], axis=0)
            + numpy.mean(testingKPIs[:,:,23,0], axis=0)
            + numpy.mean(testingKPIs[:,:,33,0], axis=0)
            + numpy.mean(testingKPIs[:,:,41,0], axis=0)
            + numpy.mean(testingKPIs[:,:,53,0], axis=0)
            ) / 6, 
        training_loss_plot2 =
            (
            numpy.mean(trainingKPIs[:,:,3,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,13,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,23,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,33,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,41,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,53,1], axis=0)
            ) / 6,
        testing_loss_plot2 =
            (
            numpy.mean(testingKPIs[:,:,3,1], axis=0)
            + numpy.mean(testingKPIs[:,:,13,1], axis=0)
            + numpy.mean(testingKPIs[:,:,23,1], axis=0)
            + numpy.mean(testingKPIs[:,:,33,1], axis=0)
            + numpy.mean(testingKPIs[:,:,41,1], axis=0)
            + numpy.mean(testingKPIs[:,:,53,1], axis=0)
            ) / 6,
        training_accuracy_plot2_standardDeviation =
            (
            numpy.std(trainingKPIs[:,:,3,0], axis=0)
            + numpy.std(trainingKPIs[:,:,13,0], axis=0)
            + numpy.std(trainingKPIs[:,:,23,0], axis=0)
            + numpy.std(trainingKPIs[:,:,33,0], axis=0)
            + numpy.std(trainingKPIs[:,:,41,0], axis=0)
            + numpy.std(trainingKPIs[:,:,53,0], axis=0)
            ) / 6,
        testing_accuracy_plot2_standardDeviation =
            (
            numpy.std(testingKPIs[:,:,3,0], axis=0)
            + numpy.std(testingKPIs[:,:,13,0], axis=0)
            + numpy.std(testingKPIs[:,:,23,0], axis=0)
            + numpy.std(testingKPIs[:,:,33,0], axis=0)
            + numpy.std(testingKPIs[:,:,41,0], axis=0)
            + numpy.std(testingKPIs[:,:,53,0], axis=0)
            ) / 6, 
        training_loss_plot2_standardDeviation =
            (
            numpy.std(trainingKPIs[:,:,3,1], axis=0)
            + numpy.std(trainingKPIs[:,:,13,1], axis=0)
            + numpy.std(trainingKPIs[:,:,23,1], axis=0)
            + numpy.std(trainingKPIs[:,:,33,1], axis=0)
            + numpy.std(trainingKPIs[:,:,41,1], axis=0)
            + numpy.std(trainingKPIs[:,:,53,1], axis=0)
            ) / 6,
        testing_loss_plot2_standardDeviation =
            (
            numpy.std(testingKPIs[:,:,3,1], axis=0)
            + numpy.std(testingKPIs[:,:,13,1], axis=0)
            + numpy.std(testingKPIs[:,:,23,1], axis=0)
            + numpy.std(testingKPIs[:,:,33,1], axis=0)
            + numpy.std(testingKPIs[:,:,41,1], axis=0)
            + numpy.std(testingKPIs[:,:,53,1], axis=0)
            ) / 6,
        # complement-based validation
        title_plot3 = "Complement",
        training_accuracy_plot3 =
            (
            numpy.mean(trainingKPIs[:,:,4,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,12,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,24,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,32,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,42,0], axis=0)
            + numpy.mean(trainingKPIs[:,:,52,0], axis=0)
            ) / 6,
        testing_accuracy_plot3 =
            (
            numpy.mean(testingKPIs[:,:,4,0], axis=0)
            + numpy.mean(testingKPIs[:,:,12,0], axis=0)
            + numpy.mean(testingKPIs[:,:,24,0], axis=0)
            + numpy.mean(testingKPIs[:,:,32,0], axis=0)
            + numpy.mean(testingKPIs[:,:,42,0], axis=0)
            + numpy.mean(testingKPIs[:,:,52,0], axis=0)
            ) / 6,
        training_loss_plot3 =
            (
            numpy.mean(trainingKPIs[:,:,4,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,12,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,24,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,32,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,42,1], axis=0)
            + numpy.mean(trainingKPIs[:,:,52,1], axis=0)
            ) / 6,
        testing_loss_plot3 =
            (
            numpy.mean(testingKPIs[:,:,4,1], axis=0)
            + numpy.mean(testingKPIs[:,:,12,1], axis=0)
            + numpy.mean(testingKPIs[:,:,24,1], axis=0)
            + numpy.mean(testingKPIs[:,:,32,1], axis=0)
            + numpy.mean(testingKPIs[:,:,42,1], axis=0)
            + numpy.mean(testingKPIs[:,:,52,1], axis=0)
            ) / 6,
        training_accuracy_plot3_standardDeviation =
            (
            numpy.std(trainingKPIs[:,:,4,0], axis=0)
            + numpy.std(trainingKPIs[:,:,12,0], axis=0)
            + numpy.std(trainingKPIs[:,:,24,0], axis=0)
            + numpy.std(trainingKPIs[:,:,32,0], axis=0)
            + numpy.std(trainingKPIs[:,:,42,0], axis=0)
            + numpy.std(trainingKPIs[:,:,52,0], axis=0)
            ) / 6,
        testing_accuracy_plot3_standardDeviation =
            (
            numpy.std(testingKPIs[:,:,4,0], axis=0)
            + numpy.std(testingKPIs[:,:,12,0], axis=0)
            + numpy.std(testingKPIs[:,:,24,0], axis=0)
            + numpy.std(testingKPIs[:,:,32,0], axis=0)
            + numpy.std(testingKPIs[:,:,42,0], axis=0)
            + numpy.std(testingKPIs[:,:,52,0], axis=0)
            ) / 6,
        training_loss_plot3_standardDeviation =
            (
            numpy.std(trainingKPIs[:,:,4,1], axis=0)
            + numpy.std(trainingKPIs[:,:,12,1], axis=0)
            + numpy.std(trainingKPIs[:,:,24,1], axis=0)
            + numpy.std(trainingKPIs[:,:,32,1], axis=0)
            + numpy.std(trainingKPIs[:,:,42,1], axis=0)
            + numpy.std(trainingKPIs[:,:,52,1], axis=0)
            ) / 6,
        testing_loss_plot3_standardDeviation =
            (
            numpy.std(testingKPIs[:,:,4,1], axis=0)
            + numpy.std(testingKPIs[:,:,12,1], axis=0)
            + numpy.std(testingKPIs[:,:,24,1], axis=0)
            + numpy.std(testingKPIs[:,:,32,1], axis=0)
            + numpy.std(testingKPIs[:,:,42,1], axis=0)
            + numpy.std(testingKPIs[:,:,52,1], axis=0)
            ) / 6,
        # baseline-based validation
        title_plot4 = "Baseline",
        training_accuracy_plot4 = (
            numpy.mean(trainingKPIs[:, :, 1, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 2, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 10, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 11, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 18, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 20, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 27, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 29, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 36, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 37, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 45, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 46, 0], axis=0)
        ) / 12,
        testing_accuracy_plot4 = (
            numpy.mean(testingKPIs[:, :, 1, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 2, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 10, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 11, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 18, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 20, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 27, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 29, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 36, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 37, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 45, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 46, 0], axis=0)
        ) / 12,
        training_loss_plot4 = (
            numpy.mean(trainingKPIs[:, :, 1, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 2, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 10, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 11, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 18, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 20, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 27, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 29, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 36, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 37, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 45, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 46, 1], axis=0)
        ) / 12,
        testing_loss_plot4 = (
            numpy.mean(testingKPIs[:, :, 1, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 2, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 10, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 11, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 18, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 20, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 27, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 29, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 36, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 37, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 45, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 46, 1], axis=0)
        ) / 12,
        training_accuracy_plot4_standardDeviation = (
            numpy.std(trainingKPIs[:, :, 1, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 2, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 10, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 11, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 18, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 20, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 27, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 29, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 36, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 37, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 45, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 46, 0], axis=0)
        ) / 12,
        testing_accuracy_plot4_standardDeviation = (
            numpy.std(testingKPIs[:, :, 1, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 2, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 10, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 11, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 18, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 20, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 27, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 29, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 36, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 37, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 45, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 46, 0], axis=0)
        ) / 12,
        training_loss_plot4_standardDeviation = (
            numpy.std(trainingKPIs[:, :, 1, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 2, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 10, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 11, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 18, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 20, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 27, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 29, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 36, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 37, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 45, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 46, 1], axis=0)
        ) / 12,
        testing_loss_plot4_standardDeviation = (
            numpy.std(testingKPIs[:, :, 1, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 2, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 10, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 11, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 18, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 20, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 27, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 29, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 36, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 37, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 45, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 46, 1], axis=0)
        ) / 12,
        # baseline_okay-based validation
        title_plot5 = "Okay Baseline",
        training_accuracy_plot5 = (
            numpy.mean(trainingKPIs[:, :, 5, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 7, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 14, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 16, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 21, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 25, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 30, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 34, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 39, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 43, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 48, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 50, 0], axis=0)
            ) / 12,
        testing_accuracy_plot5 = (
            numpy.mean(testingKPIs[:, :, 5, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 7, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 14, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 16, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 21, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 25, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 30, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 34, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 39, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 43, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 48, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 50, 0], axis=0)
            ) / 12,
        training_loss_plot5 = (
            numpy.mean(trainingKPIs[:, :, 5, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 7, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 14, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 16, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 21, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 25, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 30, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 34, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 39, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 43, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 48, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 50, 1], axis=0)
            ) / 12,
        testing_loss_plot5 = (
            numpy.mean(testingKPIs[:, :, 5, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 7, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 14, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 16, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 21, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 25, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 30, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 34, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 39, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 43, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 48, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 50, 1], axis=0)
            ) / 12,
        training_accuracy_plot5_standardDeviation = (
            numpy.std(trainingKPIs[:, :, 5, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 7, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 14, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 16, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 21, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 25, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 30, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 34, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 39, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 43, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 48, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 50, 0], axis=0)
            ) / 12,
        testing_accuracy_plot5_standardDeviation = (
            numpy.std(testingKPIs[:, :, 5, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 7, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 14, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 16, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 21, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 25, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 30, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 34, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 39, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 43, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 48, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 50, 0], axis=0)
            ) / 12,
        training_loss_plot5_standardDeviation = (
            numpy.std(trainingKPIs[:, :, 5, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 7, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 14, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 16, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 21, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 25, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 30, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 34, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 39, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 43, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 48, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 50, 1], axis=0)
            ) / 12,
        testing_loss_plot5_standardDeviation = (
            numpy.std(testingKPIs[:, :, 5, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 7, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 14, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 16, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 21, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 25, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 30, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 34, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 39, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 43, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 48, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 50, 1], axis=0)
            ) / 12,

        # baseline_defect-based validation
        title_plot6 = "Defect Baseline",
        training_accuracy_plot6 = (
            numpy.mean(trainingKPIs[:, :, 6, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 8, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 15, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 17, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 22, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 26, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 31, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 35, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 40, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 44, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 49, 0], axis=0)
            + numpy.mean(trainingKPIs[:, :, 51, 0], axis=0)
        ) / 12,
        testing_accuracy_plot6 = (
            numpy.mean(testingKPIs[:, :, 6, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 8, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 15, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 17, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 22, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 26, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 31, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 35, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 40, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 44, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 49, 0], axis=0)
            + numpy.mean(testingKPIs[:, :, 51, 0], axis=0)
        ) / 12,
        training_loss_plot6 = (
            numpy.mean(trainingKPIs[:, :, 6, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 8, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 15, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 17, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 22, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 26, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 31, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 35, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 40, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 44, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 49, 1], axis=0)
            + numpy.mean(trainingKPIs[:, :, 51, 1], axis=0)
        ) / 12,
        testing_loss_plot6 = (
            numpy.mean(testingKPIs[:, :, 6, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 8, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 15, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 17, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 22, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 26, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 31, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 35, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 40, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 44, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 49, 1], axis=0)
            + numpy.mean(testingKPIs[:, :, 51, 1], axis=0)
        ) / 12,
        training_accuracy_plot6_standardDeviation = (
            numpy.std(trainingKPIs[:, :, 6, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 8, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 15, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 17, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 22, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 26, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 31, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 35, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 40, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 44, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 49, 0], axis=0)
            + numpy.std(trainingKPIs[:, :, 51, 0], axis=0)
        ) / 12,
        testing_accuracy_plot6_standardDeviation = (
            numpy.std(testingKPIs[:, :, 6, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 8, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 15, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 17, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 22, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 26, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 31, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 35, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 40, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 44, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 49, 0], axis=0)
            + numpy.std(testingKPIs[:, :, 51, 0], axis=0)
        ) / 12,
        training_loss_plot6_standardDeviation = (
            numpy.std(trainingKPIs[:, :, 6, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 8, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 15, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 17, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 22, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 26, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 31, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 35, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 40, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 44, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 49, 1], axis=0)
            + numpy.std(trainingKPIs[:, :, 51, 1], axis=0)
        ) / 12,
        testing_loss_plot6_standardDeviation = (
            numpy.std(testingKPIs[:, :, 6, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 8, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 15, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 17, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 22, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 26, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 31, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 35, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 40, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 44, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 49, 1], axis=0)
            + numpy.std(testingKPIs[:, :, 51, 1], axis=0)
        ) / 12,
        #title_plot7, training_accuracy_plot7, testing_accuracy_plot7, training_loss_plot7, testing_loss_plot7,
        title_plot7=[], training_accuracy_plot7=[], testing_accuracy_plot7=[], training_loss_plot7=[], testing_loss_plot7=[],
        #title_plot8, training_accuracy_plot8, testing_accuracy_plot8, training_loss_plot8, testing_loss_plot8,
        title_plot8=[], training_accuracy_plot8=[], testing_accuracy_plot8=[], training_loss_plot8=[], testing_loss_plot8=[],
        #title_plot9, training_accuracy_plot9, testing_accuracy_plot9, training_loss_plot9, testing_loss_plot9,
        title_plot9=[], training_accuracy_plot9=[], testing_accuracy_plot9=[], training_loss_plot9=[], testing_loss_plot9=[],
        )

def collect_KPIs(trainingKPIs, testingKPIs, sender, dim_1, dim_2, dim_3):
    """
    This function collects KPIs from containers and copies them to docker's current build context folder.
    From here, these may be accessed and can be mapped to current KPI collection.
    The current collection is stored at log directory.
    Its individual elements can be accessed with dim_1, dim_2, dim_3 and dim_4 by:
    - trainingKPIs[experimentId-1][iterationId_1-1+iterationId_2][machineId-1+datasetId+streamId-1][kindOfKPI] = value
    - testingKPIs[experimentId-1][iterationId_1-1+iterationId_2][machineId-1+datasetId+streamId-1][kindOfKPI] = value
    - dim_4 -> 0 - accuracies, 1 - losses, 2 - n
    """

    verbose = False

    # copy KPIs to dockers current build context folder (for preparing analysis or publication to docker's hub)
    subprocess.run("docker run --rm -v $PWD/../messageClient/logs:/host -v ai_system:/ai_system -w /ai_system busybox /bin/sh -c " + "'"
                    + " cp /ai_system/"+sender+"/learningBase/training_accuracy.txt /host/"+sender+"_training_accuracy.txt" 
                    + " && cp /ai_system/"+sender+"/learningBase/training_loss.txt /host/"+sender+"_training_loss.txt"
                    + " && cp /ai_system/"+sender+"/learningBase/training_n.txt /host/"+sender+"_training_n.txt"
                    + " && cp /ai_system/"+sender+"/learningBase/testing_accuracy.txt /host/"+sender+"_testing_accuracy.txt"
                    + " && cp /ai_system/"+sender+"/learningBase/testing_loss.txt /host/"+sender+"_testing_loss.txt"
                    + " && cp /ai_system/"+sender+"/learningBase/testing_n.txt /host/"+sender+"_testing_n.txt"
                    + "'", shell=True)

    path = logDirectory + "/"+sender+"_"

    f = open(path + "training_accuracy.txt", "r")
    trainingKPIs[dim_1][dim_2][dim_3][0] = float(f.read())

    f = open(path + "training_loss.txt", "r")
    trainingKPIs[dim_1][dim_2][dim_3][1] = float(f.read())

    f = open(path + "training_n.txt", "r")
    trainingKPIs[dim_1][dim_2][dim_3][2] = float(f.read())

    f = open(path + "testing_accuracy.txt", "r")
    testingKPIs[dim_1][dim_2][dim_3][0] = float(f.read())

    f = open(path + "testing_loss.txt", "r")
    testingKPIs[dim_1][dim_2][dim_3][1] = float(f.read())

    f = open(path + "testing_n.txt", "r")
    testingKPIs[dim_1][dim_2][dim_3][2] = float(f.read())

    if verbose : print('              current dimensions:', str(dim_1), str(dim_2), str(dim_3), '    ->     Ns refer to ', str( trainingKPIs[dim_1][dim_2][dim_3][2]),  testingKPIs[dim_1][dim_2][dim_3][2])

    # store current KPI collection
    save_data_tofile(numpyArray=trainingKPIs, path=logDirectory+'/trainingKPIs.csv')
    save_data_tofile(numpyArray=testingKPIs, path=logDirectory+'/testingKPIs.csv')

    return trainingKPIs, testingKPIs

def realize_experiment():
    """
    This function realizes experiments on continual ANN training and testing on switching datasets.
    Its request arrives from communication client and it manages the corresponding AI reguests.
    """

    verbose = False

    # specify meta-parameters for experiment realization
    maxNumberOfExperiments = 10 # given by statistical requirements
    maxIterationsInPhase1 = 5   # given by experiment design
    maxIterationsInPhase2 = 5   # given by experiment design
    maxMachines = 3             # given by scenario
    maxValidationSets = 3+6     # given by experiment design (a/b/c + their variants as okay and defective)
    maxStreams = 2              # given by experiment design
    maxNumberOfKPIs = 3 # 0 - accuracies, 1 - losses, 2 - n
    
    # initialize KPI collections
    trainingKPIs = numpy.zeros((maxNumberOfExperiments, maxIterationsInPhase1+1+maxIterationsInPhase2+1, maxMachines*maxValidationSets*maxStreams, maxNumberOfKPIs))
    testingKPIs  = numpy.zeros((maxNumberOfExperiments, maxIterationsInPhase1+1+maxIterationsInPhase2+1, maxMachines*maxValidationSets*maxStreams, maxNumberOfKPIs))

    for experimentId in range(1, maxNumberOfExperiments+1, 1):
        if verbose : print("  experimentId = " + str(experimentId))
        for machineId in range(1, maxMachines+1, 1):
            if verbose : print("    machineId = " + str(machineId))

            # Phase 1 - Working on focus dataset
            # (from initial state to switching state)
            #########################################
            if verbose : print("      enterint phase 1...")
            # wire and train ANNs by refinement to create initial state (while having interim states at preparation) and publish it to docker's hub
            #do not create new ANNs, but consider initial ANNs form experiment01, so that the ANN evolution is comparable
            #aiClient.realize_scenario(scenario="wire_annSolution", knowledge_base="-", activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base="-", sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration0", receiver="ReceiverB", sub_process_method="sequential")
            #aiClient.realize_scenario(scenario="publish_annSolution", knowledge_base="-", activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base="-", sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration0", receiver="ReceiverB", sub_process_method="sequential")
            for iterationId_1 in range(1, maxIterationsInPhase1+1, 1):
                if verbose : print("        iterationId_1 = " + str(iterationId_1))
                if (machineId == 1):
                    learning_base = "marcusgrum/learningbase_apple_01"
                    if (iterationId_1 == 1):
                        suffix_0 = ""
                        suffix_1 = suffix_0 + "_a"
                    else:
                        suffix_0 = suffix_1
                        suffix_1 = suffix_1
                elif (machineId == 2):
                    learning_base = "marcusgrum/learningbase_banana_01"
                    if (iterationId_1 == 1):
                        suffix_0 = ""
                        suffix_1 = suffix_0 + "_b"
                    else:
                        suffix_0 = suffix_1
                        suffix_1 = suffix_1
                elif (machineId == 3):
                    learning_base = "marcusgrum/learningbase_orange_01"
                    if (iterationId_1 == 1):
                        suffix_0 = ""
                        suffix_1 = suffix_0 + "_o"
                    else:
                        suffix_0 = suffix_1
                        suffix_1 = suffix_1
                else:
                    pass
                # train wired ANNs by refinement to create switching state (while having interim states at preparation) and publish it to docker's hub
                #do not refine ANNs, but consider initial ANNs form experiment01, so that the ANN evolution is comparable
                #aiClient.realize_scenario(scenario="refine_annSolution", knowledge_base="marcusgrum/knowledgebase_experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1-1)+suffix_0, activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base=learning_base, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1)+suffix_1, receiver="ReceiverB", sub_process_method="sequential")
                #aiClient.realize_scenario(scenario="publish_annSolution", knowledge_base="marcusgrum/experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1)+suffix_1, activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base=learning_base, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1)+suffix_1, receiver="ReceiverB", sub_process_method="sequential")
                
                # carry out testing cases for initial testing of wired ANN
                if (iterationId_1==1):
                    for streamId in range(1, maxStreams+1, 1):
                        if verbose : print("          streamId = " + str(streamId))
                        for datasetId in range(0, maxValidationSets, 1):
                            if verbose : print("            datasetId = " + str(datasetId))
                            if (datasetId == 0):
                                sufsuffix_0 = "_evalWith_a"
                                evaluation_base="marcusgrum/learningbase_apple_01"
                            elif (datasetId == 1):
                                sufsuffix_0 = "_evalWith_b"
                                evaluation_base="marcusgrum/learningbase_banana_01"
                            elif (datasetId == 2):
                                sufsuffix_0 = "_evalWith_o"
                                evaluation_base="marcusgrum/learningbase_orange_01"
                            elif (datasetId == 3):
                                sufsuffix_0 = "_evalWith_a_okay"
                                evaluation_base="marcusgrum/learningbase_apple_okay_01"
                            elif (datasetId == 4):
                                sufsuffix_0 = "_evalWith_a_defect"
                                evaluation_base="marcusgrum/learningbase_apple_defect_01"
                            elif (datasetId == 5):
                                sufsuffix_0 = "_evalWith_b_okay"
                                evaluation_base="marcusgrum/learningbase_banana_okay_01"
                            elif (datasetId == 6):
                                sufsuffix_0 = "_evalWith_b_defect"
                                evaluation_base="marcusgrum/learningbase_banana_defect_01"
                            elif (datasetId == 7):
                                sufsuffix_0 = "_evalWith_o_okay"
                                evaluation_base="marcusgrum/learningbase_orange_okay_01"
                            elif (datasetId == 8):
                                sufsuffix_0 = "_evalWith_o_defect"
                                evaluation_base="marcusgrum/learningbase_orange_defect_01"
                            else:
                                pass
                            # evaluate wired state before first learning iteration with apple, banana and orange dataset
                            aiClient.realize_scenario(scenario="evaluate_annSolution", knowledge_base="marcusgrum/knowledgebase_experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1-1)+suffix_0, activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base=evaluation_base, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1-1)+suffix_0+sufsuffix_0, receiver="ReceiverB", sub_process_method="sequential")
                            # maybe publish evaluation containers, too?
                            # collect KPIs and take care for adequate duplication for redundant runs (e.g. phase 1 of AB and AO)
                            trainingKPIs, testingKPIs = collect_KPIs(trainingKPIs=trainingKPIs, testingKPIs=testingKPIs, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1-1)+suffix_0+sufsuffix_0, dim_1=experimentId-1, dim_2=iterationId_1-1, dim_3=((machineId-1)*maxStreams*maxValidationSets)+((streamId-1)*maxValidationSets)+datasetId)
                # carry out testing cases
                for streamId in range(1, maxStreams+1, 1):
                    if verbose : print("          streamId = " + str(streamId))
                    for datasetId in range(0, maxValidationSets, 1):
                        if verbose : print("            datasetId = " + str(datasetId))
                        if (datasetId == 0):
                            sufsuffix_0 = "_evalWith_a"
                            evaluation_base="marcusgrum/learningbase_apple_01"
                        elif (datasetId == 1):
                            sufsuffix_0 = "_evalWith_b"
                            evaluation_base="marcusgrum/learningbase_banana_01"
                        elif (datasetId == 2):
                            sufsuffix_0 = "_evalWith_o"
                            evaluation_base="marcusgrum/learningbase_orange_01"
                        elif (datasetId == 3):
                            sufsuffix_0 = "_evalWith_a_okay"
                            evaluation_base="marcusgrum/learningbase_apple_okay_01"
                        elif (datasetId == 4):
                            sufsuffix_0 = "_evalWith_a_defect"
                            evaluation_base="marcusgrum/learningbase_apple_defect_01"
                        elif (datasetId == 5):
                            sufsuffix_0 = "_evalWith_b_okay"
                            evaluation_base="marcusgrum/learningbase_banana_okay_01"
                        elif (datasetId == 6):
                            sufsuffix_0 = "_evalWith_b_defect"
                            evaluation_base="marcusgrum/learningbase_banana_defect_01"
                        elif (datasetId == 7):
                            sufsuffix_0 = "_evalWith_o_okay"
                            evaluation_base="marcusgrum/learningbase_orange_okay_01"
                        elif (datasetId == 8):
                            sufsuffix_0 = "_evalWith_o_defect"
                            evaluation_base="marcusgrum/learningbase_orange_defect_01"
                        else:
                            pass
                        # evaluate trained / refined state of current iteration with apple, banana and orange dataset
                        aiClient.realize_scenario(scenario="evaluate_annSolution", knowledge_base="marcusgrum/knowledgebase_experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1)+suffix_1, activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base=evaluation_base, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1)+suffix_1+sufsuffix_0, receiver="ReceiverB", sub_process_method="sequential")
                        # maybe publish evaluation containers, too?
                        # ...
                        # collect KPIs and take care for adequate duplication for redundant runs (e.g. phase 1 of AB and AO)
                        trainingKPIs, testingKPIs = collect_KPIs(trainingKPIs=trainingKPIs, testingKPIs=testingKPIs, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1)+suffix_1+sufsuffix_0, dim_1=experimentId-1, dim_2=iterationId_1, dim_3=((machineId-1)*maxStreams*maxValidationSets)+((streamId-1)*maxValidationSets)+datasetId)

            # Phase 2 - Working on new dataset
            # (from switching state to final state)
            #######################################

            if verbose : print("      entering phase 2...")
            for streamId in range(1, maxStreams+1, 1):
                if verbose : print("        streamId = " + str(streamId))
                for iterationId_2 in range(0, maxIterationsInPhase2+1, 1):
                    if verbose : print("          iterationId_2 = " + str(iterationId_2))
                    if (machineId == 1):
                        if (streamId == 1):
                            learning_base = "marcusgrum/learningbase_apple_okay_01"
                            if (iterationId_2 == 0):
                                suffix_2 = suffix_1
                                suffix_3 = suffix_1 + "aokay"
                            else:
                                suffix_2 = suffix_1 + "aokay"
                                suffix_3 = suffix_1 + "aokay"
                        elif (streamId == 2):
                            learning_base = "marcusgrum/learningbase_apple_defect_01"
                            if (iterationId_2 == 0):
                                suffix_2 = suffix_1
                                suffix_3 = suffix_1 + "adefect"
                            else:
                                suffix_2 = suffix_1 + "adefect"
                                suffix_3 = suffix_1 + "adefect"
                        else:
                            pass
                    elif (machineId == 2):
                        if(streamId == 1):
                            learning_base = "marcusgrum/learningbase_banana_okay_01"
                            if (iterationId_2 == 0):
                                suffix_2 = suffix_1
                                suffix_3 = suffix_1 + "bokay"
                            else:
                                suffix_2 = suffix_1 + "bokay"
                                suffix_3 = suffix_1 + "bokay"
                        elif (streamId == 2):
                            learning_base = "marcusgrum/learningbase_banana_defect_01"
                            if (iterationId_2 == 0):
                                suffix_2 = suffix_1
                                suffix_3 = suffix_1 + "bdefect"
                            else:
                                suffix_2 = suffix_1 + "bdefect"
                                suffix_3 = suffix_1 + "bdefect"
                        else:
                            pass
                    elif (machineId == 3):
                        if(streamId == 1):
                            learning_base = "marcusgrum/learningbase_orange_okay_01"
                            if (iterationId_2 == 0):
                                suffix_2 = suffix_1
                                suffix_3 = suffix_1 + "ookay"
                            else:
                                suffix_2 = suffix_1 + "ookay"
                                suffix_3 = suffix_1 + "ookay"
                        elif (streamId == 2):
                            learning_base = "marcusgrum/learningbase_orange_defect_01"
                            if (iterationId_2 == 0):
                                suffix_2 = suffix_1
                                suffix_3 = suffix_1 + "odefect"
                            else:
                                suffix_2 = suffix_1 + "odefect"
                                suffix_3 = suffix_1 + "odefect"
                        else:
                            pass
                    else:
                        pass

                    # train ANNs of phase 1 by refinement to create final state (while having interim states) and publish it to docker's hub
                    aiClient.realize_scenario(scenario="refine_annSolution", knowledge_base="marcusgrum/knowledgebase_experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1+iterationId_2)+suffix_2, activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base=learning_base, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1+iterationId_2+1)+suffix_3, receiver="ReceiverB", sub_process_method="sequential")
                    aiClient.realize_scenario(scenario="publish_annSolution", knowledge_base="marcusgrum/experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1+iterationId_2+1)+suffix_3, activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base=learning_base, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1+iterationId_2+1)+suffix_3, receiver="ReceiverB", sub_process_method="sequential")
                    
                    # carry out testing cases
                    for datasetId in range(0, maxValidationSets, 1):
                        if verbose : print("            datasetId = " + str(datasetId))
                        if (datasetId == 0):
                            sufsuffix_0 = "_evalWith_a"
                            evaluation_base="marcusgrum/learningbase_apple_01"
                        elif (datasetId == 1):
                            sufsuffix_0 = "_evalWith_b"
                            evaluation_base="marcusgrum/learningbase_banana_01"
                        elif (datasetId == 2):
                            sufsuffix_0 = "_evalWith_o"
                            evaluation_base="marcusgrum/learningbase_orange_01"
                        elif (datasetId == 3):
                            sufsuffix_0 = "_evalWith_a_okay"
                            evaluation_base="marcusgrum/learningbase_apple_okay_01"
                        elif (datasetId == 4):
                            sufsuffix_0 = "_evalWith_a_defect"
                            evaluation_base="marcusgrum/learningbase_apple_defect_01"
                        elif (datasetId == 5):
                            sufsuffix_0 = "_evalWith_b_okay"
                            evaluation_base="marcusgrum/learningbase_banana_okay_01"
                        elif (datasetId == 6):
                            sufsuffix_0 = "_evalWith_b_defect"
                            evaluation_base="marcusgrum/learningbase_banana_defect_01"
                        elif (datasetId == 7):
                            sufsuffix_0 = "_evalWith_o_okay"
                            evaluation_base="marcusgrum/learningbase_orange_okay_01"
                        elif (datasetId == 8):
                            sufsuffix_0 = "_evalWith_o_defect"
                            evaluation_base="marcusgrum/learningbase_orange_defect_01"
                        else:
                            pass
                        # evaluate trained / refined state of current iteration with apple, banana and orange dataset
                        aiClient.realize_scenario(scenario="evaluate_annSolution", knowledge_base="marcusgrum/knowledgebase_experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1+iterationId_2+1)+suffix_3, activation_base="-", code_base="marcusgrum/codebase_ai_core_for_image_classification", learning_base=evaluation_base, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1+iterationId_2+1)+suffix_3+sufsuffix_0, receiver="ReceiverB", sub_process_method="sequential")
                        # maybe publish evaluation containers, too?
                        # ...
                        # collect KPIs and take care for adequate duplication for redundant runs (e.g. phase 1 of AB and AO)
                        trainingKPIs, testingKPIs = collect_KPIs(trainingKPIs=trainingKPIs, testingKPIs=testingKPIs, sender="experiment"+str(experimentId)+"_machine"+str(machineId)+"_iteration"+str(iterationId_1+iterationId_2+1)+suffix_3+sufsuffix_0, dim_1=experimentId-1, dim_2=iterationId_1+iterationId_2+1, dim_3=((machineId-1)*maxStreams*maxValidationSets)+((streamId-1)*maxValidationSets)+datasetId)
    
    # create a visual overview over all experiment runs
    #realize_experiment_plotting(maxNumberOfExperiments = maxNumberOfExperiments, maxIterationsInPhase1 = maxIterationsInPhase1, maxIterationsInPhase2 = maxIterationsInPhase2, maxMachines = maxMachines, maxValidationSets = maxValidationSets, maxStreams = maxStreams, maxNumberOfKPIs = maxNumberOfKPIs)

def summarize_KPIs(maxNumberOfExperiments = 10, maxIterationsInPhase1 = 5, maxIterationsInPhase2 = 5, maxMachines = 3, maxValidationSets = 3, maxStreams = 2, maxNumberOfKPIs = 3):
    """
    This function separates individual KPI types for this experiment and stores each KPI summary at log directory.
    KPI types, we can find for the following types of levels:
    (-) individual plots per experiment run
        (AAok for run 1 / 2 / ... / maxNumberOfExperiments)
    (-) average plots over all experiment runs
        (AAok / AAdef / BBok / BBdef / OOok / OOdef)
    (x) overall plots averaging the same types of averaged experiment runs
        (Bias / Manipulation / Complement / Baseline / Baseline_ok / Baseline_def)
    """

    # storing KPIs
    path = logDirectory + "/Average_Over_All_Experiments_"

    # 0 - accuracies, 1 - losses, 2 - n
    trainingKPIs = load_data_fromfile(path=logDirectory+'/trainingKPIs.csv').reshape((maxNumberOfExperiments, maxIterationsInPhase1+1+maxIterationsInPhase2+1, maxMachines*maxValidationSets*maxStreams, maxNumberOfKPIs))
    testingKPIs = load_data_fromfile(path=logDirectory+'/testingKPIs.csv').reshape((maxNumberOfExperiments, maxIterationsInPhase1+1+maxIterationsInPhase2+1, maxMachines*maxValidationSets*maxStreams, maxNumberOfKPIs))

    # Build plot for sum over Bias / Manipulation / Complement / Baseline / Baseline_ok / Baseline_def
    print('separating KPIs over all biases, manipulations, complements and baselines...')
    
    # bias-based validation
    training_accuracy_bias_mu = (
        numpy.mean(trainingKPIs[:, :, 0, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 9, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 19, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 28, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 38, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 47, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_accuracy_bias_mu)], path+"training_accuracy_bias_mu")
    testing_accuracy_bias_mu = (
        numpy.mean(testingKPIs[:, :, 0, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 9, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 19, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 28, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 38, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 47, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_accuracy_bias_mu)], path+"testing_accuracy_bias_mu")
    training_loss_bias_mu = (
        numpy.mean(trainingKPIs[:, :, 0, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 9, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 19, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 28, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 38, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 47, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_loss_bias_mu)], path+"training_loss_bias_mu")
    testing_loss_bias_mu = (
        numpy.mean(testingKPIs[:, :, 0, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 9, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 19, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 28, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 38, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 47, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_loss_bias_mu)], path+"testing_loss_bias_mu")
    training_N_bias_mu = (
        numpy.mean(trainingKPIs[:, :, 0, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 9, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 19, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 28, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 38, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 47, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_N_bias_mu)], path+"training_N_bias_mu")
    testing_N_bias_mu = (
        numpy.mean(testingKPIs[:, :, 0, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 9, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 19, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 28, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 38, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 47, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_N_bias_mu)], path+"testing_N_bias_mu")
    training_accuracy_bias_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 0, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 9, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 19, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 28, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 38, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 47, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_accuracy_bias_standardDeviation)], path+"training_accuracy_bias_standardDeviation")
    testing_accuracy_bias_standardDeviation = (
        numpy.std(testingKPIs[:, :, 0, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 9, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 19, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 28, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 38, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 47, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_accuracy_bias_standardDeviation)], path+"testing_accuracy_bias_standardDeviation")
    training_loss_bias_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 0, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 9, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 19, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 28, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 38, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 47, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_loss_bias_standardDeviation)], path+"training_loss_bias_standardDeviation")
    testing_loss_bias_standardDeviation = (
        numpy.std(testingKPIs[:, :, 0, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 9, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 19, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 28, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 38, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 47, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_loss_bias_standardDeviation)], path+"testing_loss_bias_standardDeviation")
    training_N_bias_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 0, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 9, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 19, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 28, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 38, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 47, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_N_bias_standardDeviation)], path+"training_N_bias_standardDeviation")
    testing_N_bias_standardDeviation = (
        numpy.std(testingKPIs[:, :, 0, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 9, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 19, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 28, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 38, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 47, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_N_bias_standardDeviation)], path+"testing_N_bias_standardDeviation")

    # manipulation-based validation
    training_accuracy_manipulation_mu = (
        numpy.mean(trainingKPIs[:, :, 3, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 13, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 23, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 33, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 41, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 53, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_accuracy_manipulation_mu)], path+"training_accuracy_manipulation_mu")
    testing_accuracy_manipulation_mu = (
        numpy.mean(testingKPIs[:, :, 3, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 13, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 23, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 33, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 41, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 53, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_accuracy_manipulation_mu)], path+"testing_accuracy_manipulation_mu")
    training_loss_manipulation_mu = (
        numpy.mean(trainingKPIs[:, :, 3, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 13, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 23, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 33, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 41, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 53, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_loss_manipulation_mu)], path+"training_loss_manipulation_mu")
    testing_loss_manipulation_mu = (
        numpy.mean(testingKPIs[:, :, 3, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 13, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 23, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 33, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 41, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 53, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_loss_manipulation_mu)], path+"testing_loss_manipulation_mu")
    training_N_manipulation_mu = (
        numpy.mean(trainingKPIs[:, :, 3, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 13, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 23, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 33, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 41, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 53, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_N_manipulation_mu)], path+"training_N_manipulation_mu")
    testing_N_manipulation_mu = (
        numpy.mean(testingKPIs[:, :, 3, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 13, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 23, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 33, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 41, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 53, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_N_manipulation_mu)], path+"testing_N_manipulation_mu")
    training_accuracy_manipulation_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 3, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 13, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 23, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 33, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 41, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 53, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_accuracy_manipulation_standardDeviation)], path+"training_accuracy_manipulation_standardDeviation")
    testing_accuracy_manipulation_standardDeviation = (
        numpy.std(testingKPIs[:, :, 3, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 13, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 23, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 33, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 41, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 53, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_accuracy_manipulation_standardDeviation)], path+"testing_accuracy_manipulation_standardDeviation")
    training_loss_manipulation_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 3, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 13, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 23, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 33, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 41, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 53, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_loss_manipulation_standardDeviation)], path+"training_loss_manipulation_standardDeviation")
    testing_loss_manipulation_standardDeviation = (
        numpy.std(testingKPIs[:, :, 3, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 13, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 23, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 33, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 41, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 53, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_loss_manipulation_standardDeviation)], path+"testing_loss_manipulation_standardDeviation")
    training_N_manipulation_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 3, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 13, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 23, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 33, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 41, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 53, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_N_manipulation_standardDeviation)], path+"training_N_manipulation_standardDeviation")
    testing_N_manipulation_standardDeviation = (
        numpy.std(testingKPIs[:, :, 3, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 13, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 23, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 33, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 41, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 53, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_N_manipulation_standardDeviation)], path+"testing_N_manipulation_standardDeviation")
    
    # complement-based validation
    training_accuracy_complement_mu = (
        numpy.mean(trainingKPIs[:, :, 4, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 12, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 24, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 32, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 42, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 52, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_accuracy_complement_mu)], path+"training_accuracy_complement_mu")
    testing_accuracy_complement_mu = (
        numpy.mean(testingKPIs[:, :, 4, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 12, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 24, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 32, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 42, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 52, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_accuracy_complement_mu)], path+"testing_accuracy_complement_mu")
    training_loss_complement_mu = (
        numpy.mean(trainingKPIs[:, :, 4, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 12, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 24, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 32, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 42, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 52, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_loss_complement_mu)], path+"training_loss_complement_mu")
    testing_loss_complement_mu = (
        numpy.mean(testingKPIs[:, :, 4, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 12, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 24, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 32, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 42, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 52, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_loss_complement_mu)], path+"testing_loss_complement_mu")
    training_N_complement_mu = (
        numpy.mean(trainingKPIs[:, :, 4, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 12, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 24, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 32, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 42, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 52, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_N_complement_mu)], path+"training_N_complement_mu")
    testing_N_complement_mu = (
        numpy.mean(testingKPIs[:, :, 4, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 12, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 24, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 32, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 42, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 52, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_N_complement_mu)], path+"testing_N_complement_mu")
    training_accuracy_complement_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 4, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 12, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 24, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 32, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 42, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 52, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_accuracy_complement_standardDeviation)], path+"training_accuracy_complement_standardDeviation")
    testing_accuracy_complement_standardDeviation = (
        numpy.std(testingKPIs[:, :, 4, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 12, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 24, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 32, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 42, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 52, 0], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_accuracy_complement_standardDeviation)], path+"testing_accuracy_complement_standardDeviation")
    training_loss_complement_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 4, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 12, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 24, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 32, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 42, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 52, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_loss_complement_standardDeviation)], path+"training_loss_complement_standardDeviation")
    testing_loss_complement_standardDeviation = (
        numpy.std(testingKPIs[:, :, 4, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 12, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 24, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 32, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 42, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 52, 1], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_loss_complement_standardDeviation)], path+"testing_loss_complement_standardDeviation")
    training_N_complement_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 4, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 12, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 24, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 32, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 42, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 52, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(training_N_complement_standardDeviation)], path+"training_N_complement_standardDeviation")
    testing_N_complement_standardDeviation = (
        numpy.std(testingKPIs[:, :, 4, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 12, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 24, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 32, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 42, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 52, 2], axis=0)
    ) / 6
    save_data_to_CsvFile([list(testing_N_complement_standardDeviation)], path+"testing_N_complement_standardDeviation")

    # baseline-based validation
    training_accuracy_baseline_mu = (
        numpy.mean(trainingKPIs[:, :, 1, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 2, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 10, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 11, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 18, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 20, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 27, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 29, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 36, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 37, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 45, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 46, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_accuracy_baseline_mu)], path + "training_accuracy_baseline_mu")
    testing_accuracy_baseline_mu = (
        numpy.mean(testingKPIs[:, :, 1, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 2, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 10, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 11, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 18, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 20, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 27, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 29, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 36, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 37, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 45, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 46, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_accuracy_baseline_mu)], path + "testing_accuracy_baseline_mu")
    training_loss_baseline_mu = (
        numpy.mean(trainingKPIs[:, :, 1, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 2, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 10, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 11, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 18, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 20, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 27, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 29, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 36, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 37, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 45, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 46, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_loss_baseline_mu)], path + "training_loss_baseline_mu")
    testing_loss_baseline_mu = (
        numpy.mean(testingKPIs[:, :, 1, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 2, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 10, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 11, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 18, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 20, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 27, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 29, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 36, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 37, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 45, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 46, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_loss_baseline_mu)], path + "testing_loss_baseline_mu")
    training_N_baseline_mu = (
        numpy.mean(trainingKPIs[:, :, 1, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 2, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 10, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 11, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 18, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 20, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 27, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 29, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 36, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 37, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 45, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 46, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_N_baseline_mu)], path + "training_N_baseline_mu")
    testing_N_baseline_mu = (
        numpy.mean(testingKPIs[:, :, 1, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 2, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 10, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 11, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 18, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 20, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 27, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 29, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 36, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 37, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 45, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 46, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_N_baseline_mu)], path+"testing_N_baseline_mu")
    training_accuracy_baseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 1, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 2, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 10, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 11, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 18, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 20, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 27, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 29, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 36, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 37, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 45, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 46, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_accuracy_baseline_standardDeviation)], path+"training_accuracy_baseline_standardDeviation")
    testing_accuracy_baseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 1, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 2, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 10, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 11, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 18, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 20, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 27, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 29, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 36, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 37, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 45, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 46, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_accuracy_baseline_standardDeviation)], path+"testing_accuracy_baseline_standardDeviation")
    training_loss_baseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 1, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 2, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 10, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 11, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 18, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 20, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 27, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 29, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 36, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 37, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 45, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 46, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_loss_baseline_standardDeviation)], path + "training_loss_baseline_standardDeviation")
    testing_loss_baseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 1, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 2, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 10, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 11, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 18, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 20, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 27, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 29, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 36, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 37, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 45, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 46, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_loss_baseline_standardDeviation)], path + "testing_loss_baseline_standardDeviation")
    training_N_baseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 1, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 2, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 10, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 11, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 18, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 20, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 27, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 29, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 36, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 37, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 45, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 46, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_N_baseline_standardDeviation)], path + "training_N_baseline_standardDeviation")
    testing_N_baseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 1, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 2, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 10, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 11, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 18, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 20, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 27, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 29, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 36, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 37, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 45, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 46, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_N_baseline_standardDeviation)], path + "testing_N_baseline_standardDeviation")

    # okayBaseline-based validation
    training_accuracy_okayBaseline_mu = (
        numpy.mean(trainingKPIs[:, :, 5, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 7, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 14, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 16, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 21, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 25, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 30, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 34, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 39, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 43, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 48, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 50, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_accuracy_okayBaseline_mu)], path + "training_accuracy_okayBaseline_mu")
    testing_accuracy_okayBaseline_mu = (
        numpy.mean(testingKPIs[:, :, 5, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 7, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 14, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 16, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 21, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 25, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 30, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 34, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 39, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 43, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 48, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 50, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_accuracy_okayBaseline_mu)], path + "testing_accuracy_okayBaseline_mu")
    training_loss_okayBaseline_mu = (
        numpy.mean(trainingKPIs[:, :, 5, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 7, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 14, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 16, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 21, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 25, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 30, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 34, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 39, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 43, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 48, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 50, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_loss_okayBaseline_mu)], path + "training_loss_okayBaseline_mu")
    testing_loss_okayBaseline_mu = (
        numpy.mean(testingKPIs[:, :, 5, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 7, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 14, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 16, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 21, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 25, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 30, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 34, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 39, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 43, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 48, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 50, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_loss_okayBaseline_mu)], path + "testing_loss_okayBaseline_mu")
    training_N_okayBaseline_mu = (
        numpy.mean(trainingKPIs[:, :, 5, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 7, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 14, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 16, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 21, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 25, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 30, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 34, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 39, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 43, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 48, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 50, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_N_okayBaseline_mu)], path + "training_N_baseline_mu")
    testing_N_okayBaseline_mu = (
        numpy.mean(testingKPIs[:, :, 5, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 7, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 14, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 16, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 21, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 25, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 30, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 34, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 39, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 43, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 48, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 50, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_N_okayBaseline_mu)], path + "testing_N_okayBaseline_mu")
    training_accuracy_okayBaseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 5, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 7, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 14, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 16, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 21, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 25, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 30, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 34, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 39, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 43, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 48, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 50, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_accuracy_okayBaseline_standardDeviation)], path + "training_accuracy_okayBaseline_standardDeviation")
    testing_accuracy_okayBaseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 5, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 7, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 14, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 16, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 21, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 25, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 30, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 34, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 39, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 43, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 48, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 50, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_accuracy_okayBaseline_standardDeviation)], path + "testing_accuracy_okayBaseline_standardDeviation")
    training_loss_okayBaseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 5, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 7, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 14, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 16, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 21, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 25, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 30, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 34, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 39, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 43, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 48, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 50, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_loss_okayBaseline_standardDeviation)], path + "training_loss_okayBaseline_standardDeviation")
    testing_loss_okayBaseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 5, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 7, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 14, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 16, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 21, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 25, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 30, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 34, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 39, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 43, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 48, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 50, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_loss_okayBaseline_standardDeviation)], path + "testing_loss_okayBaseline_standardDeviation")
    training_N_okayBaseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 5, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 7, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 14, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 16, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 21, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 25, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 30, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 34, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 39, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 43, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 48, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 50, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_N_okayBaseline_standardDeviation)], path + "training_N_okayBaseline_standardDeviation")
    testing_N_okayBaseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 5, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 7, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 14, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 16, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 21, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 25, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 30, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 34, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 39, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 43, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 48, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 50, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_N_okayBaseline_standardDeviation)], path + "testing_N_okayBaseline_standardDeviation")

    # defectBaseline-based validation
    training_accuracy_defectBaseline_mu = (
        numpy.mean(trainingKPIs[:, :, 6, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 8, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 15, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 17, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 22, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 26, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 31, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 35, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 40, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 44, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 49, 0], axis=0)
        + numpy.mean(trainingKPIs[:, :, 51, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_accuracy_defectBaseline_mu)], path + "training_accuracy_defectBaseline_mu")
    testing_accuracy_defectBaseline_mu = (
        numpy.mean(testingKPIs[:, :, 6, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 8, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 15, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 17, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 22, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 26, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 31, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 35, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 40, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 44, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 49, 0], axis=0)
        + numpy.mean(testingKPIs[:, :, 51, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_accuracy_defectBaseline_mu)], path + "testing_accuracy_defectBaseline_mu")
    training_loss_defectBaseline_mu = (
        numpy.mean(trainingKPIs[:, :, 6, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 8, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 15, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 17, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 22, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 26, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 31, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 35, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 40, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 44, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 49, 1], axis=0)
        + numpy.mean(trainingKPIs[:, :, 51, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_loss_defectBaseline_mu)], path + "training_loss_defectBaseline_mu")
    testing_loss_defectBaseline_mu = (
        numpy.mean(testingKPIs[:, :, 6, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 8, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 15, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 17, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 22, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 26, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 31, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 35, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 40, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 44, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 49, 1], axis=0)
        + numpy.mean(testingKPIs[:, :, 51, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_loss_defectBaseline_mu)], path + "testing_loss_defectBaseline_mu")
    training_N_defectBaseline_mu = (
        numpy.mean(trainingKPIs[:, :, 6, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 8, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 15, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 17, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 22, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 26, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 31, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 35, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 40, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 44, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 49, 2], axis=0)
        + numpy.mean(trainingKPIs[:, :, 51, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_N_defectBaseline_mu)], path + "training_N_baseline_mu")
    testing_N_defectBaseline_mu = (
        numpy.mean(testingKPIs[:, :, 6, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 8, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 15, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 17, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 22, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 26, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 31, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 35, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 40, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 44, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 49, 2], axis=0)
        + numpy.mean(testingKPIs[:, :, 51, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_N_defectBaseline_mu)], path + "testing_N_defectBaseline_mu")
    training_accuracy_defectBaseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 6, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 8, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 15, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 17, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 22, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 26, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 31, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 35, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 40, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 44, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 49, 0], axis=0)
        + numpy.std(trainingKPIs[:, :, 51, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_accuracy_defectBaseline_standardDeviation)], path + "training_accuracy_defectBaseline_standardDeviation")
    testing_accuracy_defectBaseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 6, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 8, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 15, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 17, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 22, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 26, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 31, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 35, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 40, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 44, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 49, 0], axis=0)
        + numpy.std(testingKPIs[:, :, 51, 0], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_accuracy_defectBaseline_standardDeviation)], path + "testing_accuracy_defectBaseline_standardDeviation")
    training_loss_defectBaseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 6, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 8, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 15, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 17, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 22, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 26, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 31, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 35, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 40, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 44, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 49, 1], axis=0)
        + numpy.std(trainingKPIs[:, :, 51, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_loss_defectBaseline_standardDeviation)], path + "training_loss_defectBaseline_standardDeviation")
    testing_loss_defectBaseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 6, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 8, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 15, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 17, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 22, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 26, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 31, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 35, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 40, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 44, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 49, 1], axis=0)
        + numpy.std(testingKPIs[:, :, 51, 1], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_loss_defectBaseline_standardDeviation)], path + "testing_loss_defectBaseline_standardDeviation")
    training_N_defectBaseline_standardDeviation = (
        numpy.std(trainingKPIs[:, :, 6, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 8, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 15, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 17, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 22, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 26, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 31, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 35, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 40, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 44, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 49, 2], axis=0)
        + numpy.std(trainingKPIs[:, :, 51, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(training_N_defectBaseline_standardDeviation)], path + "training_N_defectBaseline_standardDeviation")
    testing_N_defectBaseline_standardDeviation = (
        numpy.std(testingKPIs[:, :, 6, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 8, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 15, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 17, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 22, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 26, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 31, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 35, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 40, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 44, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 49, 2], axis=0)
        + numpy.std(testingKPIs[:, :, 51, 2], axis=0)
    ) / 12
    save_data_to_CsvFile([list(testing_N_defectBaseline_standardDeviation)], path + "testing_N_defectBaseline_standardDeviation")

    # Overview-specific KPIs
    rows = len(testing_N_baseline_standardDeviation)
    columns = 3 * 6 # (mu, std, n) * (bias, manipulation, complement, baseline, baseline_okay, baseline_defect)
    decimals = 3
    training_overview_accuracy = numpy.zeros((rows,columns))
    training_overview_loss = numpy.zeros((rows,columns))
    testing_overview_accuracy = numpy.zeros((rows,columns))
    testing_overview_loss = numpy.zeros((rows,columns))
    for i in range(rows):
        # create training overview in terms of accuracy
        training_overview_accuracy[i,0]  = numpy.around(training_accuracy_bias_mu[i], decimals = decimals)
        training_overview_accuracy[i,1]  = numpy.around(training_accuracy_bias_standardDeviation[i], decimals = decimals)
        training_overview_accuracy[i,2]  = numpy.around(training_N_bias_mu[i], decimals = decimals)
        training_overview_accuracy[i,3]  = numpy.around(training_accuracy_manipulation_mu[i], decimals = decimals)
        training_overview_accuracy[i,4]  = numpy.around(training_accuracy_manipulation_standardDeviation[i], decimals = decimals)
        training_overview_accuracy[i,5]  = numpy.around(training_N_manipulation_mu[i], decimals = decimals)
        training_overview_accuracy[i,6]  = numpy.around(training_accuracy_complement_mu[i], decimals = decimals)
        training_overview_accuracy[i,7]  = numpy.around(training_accuracy_complement_standardDeviation[i], decimals = decimals)
        training_overview_accuracy[i,8]  = numpy.around(training_N_complement_mu[i], decimals = decimals)
        training_overview_accuracy[i,9]  = numpy.around(training_accuracy_baseline_mu[i], decimals = decimals)
        training_overview_accuracy[i,10] = numpy.around(training_accuracy_baseline_standardDeviation[i], decimals = decimals)
        training_overview_accuracy[i,11] = numpy.around(training_N_baseline_mu[i], decimals = decimals)
        training_overview_accuracy[i,12] = numpy.around(training_accuracy_okayBaseline_mu[i], decimals = decimals)
        training_overview_accuracy[i,13] = numpy.around(training_accuracy_okayBaseline_standardDeviation[i], decimals = decimals)
        training_overview_accuracy[i,14] = numpy.around(training_N_okayBaseline_mu[i], decimals = decimals)
        training_overview_accuracy[i,15] = numpy.around(training_accuracy_defectBaseline_mu[i], decimals = decimals)
        training_overview_accuracy[i,16] = numpy.around(training_accuracy_defectBaseline_standardDeviation[i], decimals = decimals)
        training_overview_accuracy[i,17] = numpy.around(training_N_defectBaseline_mu[i], decimals = decimals)

        # create training overview in terms of loss
        training_overview_loss[i,0]  = numpy.around(training_loss_bias_mu[i], decimals = decimals)
        training_overview_loss[i,1]  = numpy.around(training_loss_bias_standardDeviation[i], decimals = decimals)
        training_overview_loss[i,2]  = numpy.around(training_N_bias_mu[i], decimals = decimals)
        training_overview_loss[i,3]  = numpy.around(training_loss_manipulation_mu[i], decimals = decimals)
        training_overview_loss[i,4]  = numpy.around(training_loss_manipulation_standardDeviation[i], decimals = decimals)
        training_overview_loss[i,5]  = numpy.around(training_N_manipulation_mu[i], decimals = decimals)
        training_overview_loss[i,6]  = numpy.around(training_loss_complement_mu[i], decimals = decimals)
        training_overview_loss[i,7]  = numpy.around(training_loss_complement_standardDeviation[i], decimals = decimals)
        training_overview_loss[i,8]  = numpy.around(training_N_complement_mu[i], decimals = decimals)
        training_overview_loss[i,9]  = numpy.around(training_loss_baseline_mu[i], decimals = decimals)
        training_overview_loss[i,10] = numpy.around(training_loss_baseline_standardDeviation[i], decimals = decimals)
        training_overview_loss[i,11] = numpy.around(training_N_baseline_mu[i], decimals = decimals)
        training_overview_loss[i,12] = numpy.around(training_loss_okayBaseline_mu[i], decimals = decimals)
        training_overview_loss[i,13] = numpy.around(training_loss_okayBaseline_standardDeviation[i], decimals = decimals)
        training_overview_loss[i,14] = numpy.around(training_N_okayBaseline_mu[i], decimals = decimals)
        training_overview_loss[i,15] = numpy.around(training_loss_defectBaseline_mu[i], decimals = decimals)
        training_overview_loss[i,16] = numpy.around(training_loss_defectBaseline_standardDeviation[i], decimals = decimals)
        training_overview_loss[i,17] = numpy.around(training_N_defectBaseline_mu[i], decimals = decimals)

        # create testing overview in terms of accuracy
        testing_overview_accuracy[i,0]  = numpy.around(testing_accuracy_bias_mu[i], decimals = decimals)
        testing_overview_accuracy[i,1]  = numpy.around(testing_accuracy_bias_standardDeviation[i], decimals = decimals)
        testing_overview_accuracy[i,2]  = numpy.around(testing_N_bias_mu[i], decimals = decimals)
        testing_overview_accuracy[i,3]  = numpy.around(testing_accuracy_manipulation_mu[i], decimals = decimals)
        testing_overview_accuracy[i,4]  = numpy.around(testing_accuracy_manipulation_standardDeviation[i], decimals = decimals)
        testing_overview_accuracy[i,5]  = numpy.around(testing_N_manipulation_mu[i], decimals = decimals)
        testing_overview_accuracy[i,6]  = numpy.around(testing_accuracy_complement_mu[i], decimals = decimals)
        testing_overview_accuracy[i,7]  = numpy.around(testing_accuracy_complement_standardDeviation[i], decimals = decimals)
        testing_overview_accuracy[i,8]  = numpy.around(testing_N_complement_mu[i], decimals = decimals)
        testing_overview_accuracy[i,9]  = numpy.around(testing_accuracy_baseline_mu[i], decimals = decimals)
        testing_overview_accuracy[i,10] = numpy.around(testing_accuracy_baseline_standardDeviation[i], decimals = decimals)
        testing_overview_accuracy[i,11] = numpy.around(testing_N_baseline_mu[i], decimals = decimals)
        testing_overview_accuracy[i,12] = numpy.around(testing_accuracy_okayBaseline_mu[i], decimals = decimals)
        testing_overview_accuracy[i,13] = numpy.around(testing_accuracy_okayBaseline_standardDeviation[i], decimals = decimals)
        testing_overview_accuracy[i,14] = numpy.around(testing_N_okayBaseline_mu[i], decimals = decimals)
        testing_overview_accuracy[i,15] = numpy.around(testing_accuracy_defectBaseline_mu[i], decimals = decimals)
        testing_overview_accuracy[i,16] = numpy.around(testing_accuracy_defectBaseline_standardDeviation[i], decimals = decimals)
        testing_overview_accuracy[i,17] = numpy.around(testing_N_defectBaseline_mu[i], decimals = decimals)

        # create testing overview in terms of loss
        testing_overview_loss[i,0]  = numpy.around(testing_loss_bias_mu[i], decimals = decimals)
        testing_overview_loss[i,1]  = numpy.around(testing_loss_bias_standardDeviation[i], decimals = decimals)
        testing_overview_loss[i,2]  = numpy.around(testing_N_bias_mu[i], decimals = decimals)
        testing_overview_loss[i,3]  = numpy.around(testing_loss_manipulation_mu[i], decimals = decimals)
        testing_overview_loss[i,4]  = numpy.around(testing_loss_manipulation_standardDeviation[i], decimals = decimals)
        testing_overview_loss[i,5]  = numpy.around(testing_N_manipulation_mu[i], decimals = decimals)
        testing_overview_loss[i,6]  = numpy.around(testing_loss_complement_mu[i], decimals = decimals)
        testing_overview_loss[i,7]  = numpy.around(testing_loss_complement_standardDeviation[i], decimals = decimals)
        testing_overview_loss[i,8]  = numpy.around(testing_N_complement_mu[i], decimals = decimals)
        testing_overview_loss[i,9]  = numpy.around(testing_loss_baseline_mu[i], decimals = decimals)
        testing_overview_loss[i,10] = numpy.around(testing_loss_baseline_standardDeviation[i], decimals = decimals)
        testing_overview_loss[i,11] = numpy.around(testing_N_baseline_mu[i], decimals = decimals)
        testing_overview_loss[i,12] = numpy.around(testing_loss_okayBaseline_mu[i], decimals = decimals)
        testing_overview_loss[i,13] = numpy.around(testing_loss_okayBaseline_standardDeviation[i], decimals = decimals)
        testing_overview_loss[i,14] = numpy.around(testing_N_okayBaseline_mu[i], decimals = decimals)
        testing_overview_loss[i,15] = numpy.around(testing_loss_defectBaseline_mu[i], decimals = decimals)
        testing_overview_loss[i,16] = numpy.around(testing_loss_defectBaseline_standardDeviation[i], decimals = decimals)
        testing_overview_loss[i,17] = numpy.around(testing_N_defectBaseline_mu[i], decimals = decimals)

    # store overview KPIs
    save_data_to_CsvFile([list(training_overview_accuracy)][0], path+"training_overview_accuracy")
    save_data_to_CsvFile([list(training_overview_loss)][0], path+"training_overview_loss")
    save_data_to_CsvFile([list(testing_overview_accuracy)][0], path+"testing_overview_accuracy")
    save_data_to_CsvFile([list(testing_overview_loss)][0], path+"testing_overview_loss")

if __name__ == '__main__':
    
    # when script is started manually, initiate experiment incl. plotting
    realize_experiment()

    # when script is started manually, initiate plotting
    #realize_experiment_plotting(maxNumberOfExperiments = 10, maxIterationsInPhase1 = 5, maxIterationsInPhase2 = 5, maxMachines = 3, maxValidationSets = 9, maxStreams = 2, maxNumberOfKPIs = 3)

    # when script is started manually, initiate KPI separation for storing in individual files
    #summarize_KPIs(maxNumberOfExperiments = 10, maxIterationsInPhase1 = 5, maxIterationsInPhase2 = 5, maxMachines = 3, maxValidationSets = 9, maxStreams = 2, maxNumberOfKPIs = 3)

    # comment out to keep current results and avoid accidental activation
    pass