"""
A little script to serve as communication client within the CoNM environment.
When requested, it initiates the corresponding AI activation via docker compose files.
Copyright (c) 2022 Marcus Grum
"""

__author__ = 'Marcus Grum, marcus.grum@uni-potsdam.de'
# SPDX-License-Identifier: AGPL-3.0-or-later or individual license
# SPDX-FileCopyrightText: 2022 Marcus Grum <marcus.grum@uni-potsdam.de>

import subprocess
import paho.mqtt.client as mqtt
from multiprocessing import Process, Queue, current_process, freeze_support
import time
import csv
import os
import platform
import numpy

# import experiments
import sys
sys.path.insert(0, '../experiments')
import experiment01, experiment02, experiment03, experiment04, experiment05

# specify global variables, so that they are known (1) at messageClient start and (2) at function calls from external scripts
global hostName, hostArch, logDirectory
hostName = os.uname()[1]
hostArch = platform.machine()
logDirectory = "./logs"  # = $PWD/logs
try:
    subprocess.check_output('nvidia-smi')
    print('Nvidia GPU detected!')
    hostArch = hostArch + "_gpu"
except Exception:
    print('No Nvidia GPU in system!')
    hostArch = hostArch + ""
if not os.path.exists(logDirectory):
    os.makedirs(logDirectory)

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

# The callback for when the client receives a CONNACK response from the server.


def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe(MQTT_Topic_CoNM, qos=0)  # channel to deal with CoNM
    # ...

# The callback for when a PUBLISH message is received from the server.


def on_message(client, userdata, msg):
     """
     This function continuously receives messages from broker and starts scenario realization.
     It can be called via the following CLI commands:
          1. Initiate example apply_annSolution from remote (for image classification):
          mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=apply_annSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_20, activation_base=marcusgrum/activationbase_apple_okay_01, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
          2. Initiate example create_annSolution from remote (for image classification):
          mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=create_annSolution, knowledge_base=-, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
          3. Initiate example refine_annSolution from remote (for image classification):
          mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=refine_annSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_01, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
          4. Initiate example wire_annSolution from remote (for image classification):
          mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=wire_annSolution, knowledge_base=-, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
          5. Initiate example publish_annSolution from remote (for image classification):
          mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=publish_annSolution, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
          6. Initiate experiment realize_annExperiment from remote:
          mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=realize_annExperiment, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
          
          1b. Initiate example apply_annSolution_for_transportClassification from remote (for transport classification):
          mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=apply_annSolution_for_transportClassification, knowledge_base=marcusgrum/knowledgebase_cps1_transport_system_01, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_transport_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
     """

     # provide variables as global so that these are known in this thread
     # global hostName

     # unroll messages
     message = msg.payload.decode()
     topic = msg.topic
     print(msg.topic + " " + str(message))
     scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver = unroll_message(str(message))

     if(receiver == hostName):
          # realize scenario, such as create_annSolution / apply_annSolution / refine_annSolution / publish_annSolution #/ realize_annExperiment
          realize_scenario(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver, sub_process_method="parallel")

          print('Message of ' + sender + ' has been initiated at ' + receiver + ' by ' + hostName + '(' + os.uname()[1] + ') successfully!')


def unroll_message(message):
     """
     This functions unrolls variables from message and returns them.
     """

     scenario = (message.partition("scenario=")[2]).partition(", knowledge_base=")[0]
     knowledge_base = (message.partition("knowledge_base=")[2]).partition(", activation_base=")[0]
     activation_base = (message.partition("activation_base=")[2]).partition(", code_base=")[0]
     code_base = (message.partition("code_base=")[2]).partition(", learning_base=")[0]
     learning_base = (message.partition("learning_base=")[2]).partition(", sender=")[0]
     sender = (message.partition("sender=")[2]).partition(", receiver=")[0]
     receiver = (message.partition("receiver=")[2]).partition(".")[0]

     return scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver

def unroll_sensorValuesFromScenario(message):
     """
     This functions unrolls variables from message and returns them.
     """

     scenario                            = message.partition(", cps1_conveyor_workpieceSensorLeft=")[0]
     cps1_conveyor_workpieceSensorLeft   = (message.partition("cps1_conveyor_workpieceSensorLeft=")[2]).partition(", cps1_conveyor_workpieceSensorCenter=")[0]
     cps1_conveyor_workpieceSensorCenter = (message.partition("cps1_conveyor_workpieceSensorCenter=")[2]).partition(", cps1_conveyor_workpieceSensorRight=")[0]
     cps1_conveyor_workpieceSensorRight  = (message.partition("cps1_conveyor_workpieceSensorRight=")[2]).partition(", cps2_conveyor_workpieceSensorLeft=")[0]
     cps2_conveyor_workpieceSensorLeft   = (message.partition("cps2_conveyor_workpieceSensorLeft=")[2]).partition(", cps2_conveyor_workpieceSensorCenter=")[0]
     cps2_conveyor_workpieceSensorCenter = (message.partition("cps2_conveyor_workpieceSensorCenter=")[2]).partition(", cps2_conveyor_workpieceSensorRight=")[0]
     cps2_conveyor_workpieceSensorRight  = (message.partition("cps2_conveyor_workpieceSensorRight=")[2]).partition(", knowledge_base=")[0]
     
     return scenario, cps1_conveyor_workpieceSensorLeft, cps1_conveyor_workpieceSensorCenter, cps1_conveyor_workpieceSensorRight, cps2_conveyor_workpieceSensorLeft, cps2_conveyor_workpieceSensorCenter, cps2_conveyor_workpieceSensorRight

def build_docker_file_for_publication_at_dockerhub(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
     """
     This functions builds docker file for ANN storage at Docker's hub.
     The file is stored at current working directory.
     - So far working for publishing knowledgeBase
     - Might be extended for publishing activationBase, codeBase and learningBase if required.
     """

     # if architecture = 'x86_64'
     if (hostArch == 'aarch64') or (hostArch == 'x86_64') or  (hostArch == 'x86_64_gpu'):
          with open(logDirectory+'/'+sender+'-docker-file', 'w') as f:
               f.write('# syntax=docker/dockerfile:1'+'\n')
               f.write('FROM busybox'+'\n')
               f.write(
                    'ADD ./'+sender+'_currentSolution.h5  /knowledgeBase/currentSolution.h5'+'\n')

def build_docker_compose_file_for_apply_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
     """
     This functions builds docker-compose file for scenario called apply_annSolution
     and considers variables from message, here.
     The file is stored at current working directory.
     """

     # if architecture = 'x86_64'
     if (hostArch == 'x86_64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.0"'+'\n')
               f.write('services:'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_20
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  activation_base_'+sender+':\n') # e.g. marcusgrum/activationbase_apple_okay_01
               f.write('    image: ' + activation_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/activationBase/ && mkdir -p /tmp/' + sender+'/activationBase/ && cp -r /activationBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('      - "activation_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/apply_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'x86_64_gpu'
     if (hostArch == 'x86_64_gpu'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
               f.write('services:'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_20
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  activation_base_'+sender+':\n') # e.g. marcusgrum/activationbase_apple_okay_01
               f.write('    image: ' + activation_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/activationBase/ && mkdir -p /tmp/' + sender+'/activationBase/ && cp -r /activationBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
               f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
               f.write('    # daemon.json.'+'\n')
               f.write('    runtime: nvidia'+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('      - "activation_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/apply_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'aarch64'
     if (hostArch == 'aarch64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.9"'+'\n')
               f.write('services:'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_20
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  activation_base_'+sender+':\n') # e.g. marcusgrum/activationbase_apple_okay_01
               f.write('    image: ' + activation_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/activationBase/ && mkdir -p /tmp/' + sender+'/activationBase/ && cp -r /activationBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n')
               f.write('    user: root'+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('      - "activation_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/apply_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

def build_docker_compose_file_for_apply_annSolution_of_transportClassification(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
     """
     This functions builds docker-compose file for scenario called apply_annSolution
     and considers variables from message, here.
     The file is stored at current working directory.
     """

     # if architecture = 'x86_64' or if architecture = 'x86_64_gpu'
     if (hostArch == 'x86_64') or (hostArch == 'x86_64_gpu'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.0"'+'\n')
               f.write('services:'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_cps1_transport_system_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               # no activation base since activation base of previous ann activation is used
               #f.write('  activation_base_'+sender+':\n') # e.g. marcusgrum/activationbase_apple_okay_01
               #f.write('    image: ' + activation_base + ''+'\n')
               #f.write('    volumes:'+'\n')
               #f.write('       - ai_system:/tmp/'+''+'\n')
               #f.write('    command:'+'\n')
               #f.write('    - sh'+'\n')
               #f.write('    - "-c"'+'\n')
               #f.write('    - |'+'\n')
               #f.write('      rm -rf /tmp/'+sender+'/activationBase/ && mkdir -p /tmp/' + sender+'/activationBase/ && cp -r /activationBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_transport_classification_x86_64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               #f.write('      - "activation_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/apply_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'aarch64'
     if (hostArch == 'aarch64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.9"'+'\n')
               f.write('services:'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_cps1_transport_system_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               # no activation base since activation base of previous ann activation is used
               #f.write('  activation_base_'+sender+':\n') # e.g. marcusgrum/activationbase_apple_okay_01
               #f.write('    image: ' + activation_base + ''+'\n')
               #f.write('    volumes:'+'\n')
               #f.write('       - ai_system:/tmp/'+''+'\n')
               #f.write('    command:'+'\n')
               #f.write('    - sh'+'\n')
               #f.write('    - "-c"'+'\n')
               #f.write('    - |'+'\n')
               #f.write('      rm -rf /tmp/'+sender+'/activationBase/ && mkdir -p /tmp/' + sender+'/activationBase/ && cp -r /activationBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n')
               f.write('    user: root'+'\n') # e.g. marcusgrum/codebase_ai_core_for_transport_classification_x86_64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               #f.write('      - "activation_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/apply_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

def build_docker_compose_file_for_create_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
     """
     This functions builds docker-compose file for scenario called create_annSolution
     and considers variables from message, here.
     The file is stored at current working directory.
     """

     # if architecture = 'x86_64'
     if (hostArch == 'x86_64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.0"'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/create_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'x86_64_gpu'
     if (hostArch == 'x86_64_gpu'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
               f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
               f.write('    # daemon.json.'+'\n')
               f.write('    runtime: nvidia'+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/create_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'aarch64'
     if (hostArch == 'aarch64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.9"'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n')
               f.write('    user: root'+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/create_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

def build_docker_compose_file_for_refine_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
     """
     This functions builds docker-compose file for scenario called refine_annSolution
     and considers variables from message, here.
     The file is stored at current working directory.
     """

     # if architecture = 'x86_64'
     if (hostArch == 'x86_64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.0"'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/refine_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'x86_64_gpu'
     if (hostArch == 'x86_64_gpu'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
               f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
               f.write('    # daemon.json.'+'\n')
               f.write('    runtime: nvidia'+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/refine_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'aarch64'
     if (hostArch == 'aarch64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.9"'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n')
               f.write('    user: root'+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/refine_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

def build_docker_compose_file_for_evaluate_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
     """
     This functions builds docker-compose file for scenario called evaluate_annSolution
     and considers variables from message, here.
     The file is stored at current working directory.
     """

     # if architecture = 'x86_64'
     if (hostArch == 'x86_64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.0"'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/evaluate_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'x86_64_gpu'
     if (hostArch == 'x86_64_gpu'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
               f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
               f.write('    # daemon.json.'+'\n')
               f.write('    runtime: nvidia'+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/evaluate_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'aarch64'
     if (hostArch == 'aarch64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.9"'+'\n')
               f.write('services:'+'\n')
               f.write('  learning_base_'+sender+':\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
               f.write('    image: ' + learning_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/learningBase/ && mkdir -p /tmp/' + sender+'/learningBase/ && cp -r /learningBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  knowledge_base_'+sender+':\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
               f.write('    image: ' + knowledge_base + ''+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/knowledgeBase/ && mkdir -p /tmp/' + sender+'/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('  code_base_'+sender+':\n')
               f.write('    user: root'+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    depends_on:'+'\n')
               f.write('      - "learning_base_'+sender+'"'+'\n')
               f.write('      - "knowledge_base_'+sender+'"'+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/evaluate_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

def build_docker_compose_file_for_wire_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
     """
     This functions builds docker-compose file for scenario called wire_annSolution
     and considers variables from message, here.
     The file is stored at current working directory.
     """

     # if architecture = 'x86_64'
     if (hostArch == 'x86_64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.0"'+'\n')
               f.write('services:'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/wire_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'x86_64_gpu'
     if (hostArch == 'x86_64_gpu'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
               f.write('services:'+'\n')
               f.write('  code_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
               f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
               f.write('    # daemon.json.'+'\n')
               f.write('    runtime: nvidia'+'\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/wire_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

     # if architecture = 'aarch64'
     if (hostArch == 'aarch64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.9"'+'\n')
               f.write('services:'+'\n')
               f.write('  code_base_'+sender+':\n')
               f.write('    user: root'+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
               f.write('    image: ' + code_base + '_' + hostArch + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      rm -rf /tmp/'+sender+'/codeBase/ && mkdir -p /tmp/' + sender+'/codeBase/ && cp -r /codeBase/ /tmp/'+sender+'/;'+'\n')
               f.write('      python3 /tmp/'+sender + '/codeBase/wire_annSolution.py ' + sender + " " + receiver + ';\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

def build_docker_compose_file_for_manual_sensorValueUpdate(sender, cps1_conveyor_workpieceSensorLeft, cps1_conveyor_workpieceSensorCenter, cps1_conveyor_workpieceSensorRight, cps2_conveyor_workpieceSensorLeft, cps2_conveyor_workpieceSensorCenter, cps2_conveyor_workpieceSensorRight):
     """
     This functions builds docker-compose file for scenario called manual_sensorValueUpdate
     and considers variables from message, here.
     The file is stored at current working directory.
     """

     # if architecture = 'x86_64' or if architecture = 'x86_64_gpu' or if architecture = 'aarch64'
     if (hostArch == 'x86_64') or (hostArch == 'x86_64_gpu') or (hostArch == 'aarch64'):
          with open(logDirectory+'/'+sender+'-docker-compose.yml', 'w') as f:
               f.write('version: "3.0"'+'\n')
               f.write('services:'+'\n')
               f.write('  sensor_base_'+sender+':\n') # e.g. marcusgrum/codebase_ai_core_for_transport_classification_x86_64
               f.write('    image: busybox' + '\n')
               f.write('    volumes:'+'\n')
               f.write('       - ai_system:/tmp/'+''+'\n')
               f.write('    command:'+'\n')
               f.write('    - sh'+'\n')
               f.write('    - "-c"'+'\n')
               f.write('    - |'+'\n')
               f.write('      mkdir -p /tmp/' + sender + '/activationBase/currentActivation/;\n')
               f.write('      echo ' + cps1_conveyor_workpieceSensorLeft   + ' > /tmp/' + sender + '/activationBase/currentActivation/cps1_conveyor_workpieceSensorLeft.txt;\n')
               f.write('      echo ' + cps1_conveyor_workpieceSensorCenter + ' > /tmp/' + sender + '/activationBase/currentActivation/cps1_conveyor_workpieceSensorCenter.txt;\n')
               f.write('      echo ' + cps1_conveyor_workpieceSensorRight  + ' > /tmp/' + sender + '/activationBase/currentActivation/cps1_conveyor_workpieceSensorRight.txt;\n')
               f.write('      echo ' + cps2_conveyor_workpieceSensorLeft   + ' > /tmp/' + sender + '/activationBase/currentActivation/cps2_conveyor_workpieceSensorLeft.txt;\n')
               f.write('      echo ' + cps2_conveyor_workpieceSensorCenter + ' > /tmp/' + sender + '/activationBase/currentActivation/cps2_conveyor_workpieceSensorCenter.txt;\n')
               f.write('      echo ' + cps2_conveyor_workpieceSensorRight  + ' > /tmp/' + sender + '/activationBase/currentActivation/cps2_conveyor_workpieceSensorRight.txt;\n')
               f.write('volumes:'+'\n')
               f.write('  ai_system:'+'\n')
               f.write('    external: true'+'\n')

def realize_scenario(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver, sub_process_method):
     """
     This function realizes scenarios, such as from communication client
     and manages the corresponding AI reguests.
     """

     # build docker-compose file based on message
     # for standard situations (experiment01-04)
     if (scenario == 'apply_annSolution'):
          build_docker_compose_file_for_apply_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
     if (scenario == 'create_annSolution'):
          build_docker_compose_file_for_create_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
     if (scenario == 'refine_annSolution'):
          build_docker_compose_file_for_refine_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
     if (scenario == 'wire_annSolution'):
          build_docker_compose_file_for_wire_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
     if (scenario == 'evaluate_annSolution'):
          build_docker_compose_file_for_evaluate_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)

     # for experiment05
     if (scenario == 'apply_annSolution_for_imageClassification'):
          build_docker_compose_file_for_apply_annSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
     if (scenario == 'apply_annSolution_for_transportClassification'):
          build_docker_compose_file_for_apply_annSolution_of_transportClassification(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
     if ('manual_sensorValueUpdate' in scenario):
          scenario, cps1_conveyor_workpieceSensorLeft, cps1_conveyor_workpieceSensorCenter, cps1_conveyor_workpieceSensorRight, cps2_conveyor_workpieceSensorLeft, cps2_conveyor_workpieceSensorCenter, cps2_conveyor_workpieceSensorRight = unroll_sensorValuesFromScenario(scenario)
          build_docker_compose_file_for_manual_sensorValueUpdate(sender, cps1_conveyor_workpieceSensorLeft, cps1_conveyor_workpieceSensorCenter, cps1_conveyor_workpieceSensorRight, cps2_conveyor_workpieceSensorLeft, cps2_conveyor_workpieceSensorCenter, cps2_conveyor_workpieceSensorRight)

     # realize instructions from messages by running docker-compose file created at machine-specific working directory
     if (scenario == 'apply_annSolution') or (scenario == 'create_annSolution') or (scenario == 'refine_annSolution') or (scenario == 'wire_annSolution') or (scenario == 'evaluate_annSolution'):

          if (sub_process_method == "sequential"):
               # a) by subprocess.run() or by subprocess.call() [depreciated]
               # Remark: By this variant, parallel requests at the same machine are realized sequentially, which is managed by message broaker (next request is delivered when previous request has been finished).
               #         So, requests are realized one after the other.
               # subprocess.call("docker-compose -f "+logDirectory+"/"+sender+"-docker-compose.yml up --remove-orphans", shell=True)
               subprocess.run("docker-compose -f "+logDirectory+"/"+sender+"-docker-compose.yml up --remove-orphans", shell=True)
               # print('Message of ' + sender + ' has been processed at ' + receiver + ' successfully!')

          if (sub_process_method == "parallel"):
               # b) by subprocess.Popen()
               # Remark: By this variant, parallel requests at the same machine are realized in parallel. Hence, individual stdout and stderr have been created so that CLI output is separated correctly.
               # Please note, message broaker does not manage requests. Indeed, each machine requires a manager for efficient ressource allocation.
               with open(logDirectory+"/"+sender+"_stdout.txt", "wb") as out, open(logDirectory+"/"+sender+"_stderr.txt", "wb") as err:
                    # carry out current scenario
                    p = subprocess.Popen("docker-compose -f "+logDirectory+"/"+sender+"-docker-compose.yml up --remove-orphans", shell=True, stdout=out, stderr=err)
                    print('Message of ' + sender + ' has been triggered at ' + receiver + ' successfully!')

     # If new knowledgeBase needs to be published to docker's hub, when create or refine scenarios have been finalized:
     if (scenario == 'publish_annSolution'):
          # 1. specify docker file for knowledgeBase (for preparing publication to docker's hub)
          build_docker_file_for_publication_at_dockerhub(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)

          if (sub_process_method == "sequential"):
               # 2. copy ANN to dockers current build context folder (for preparing publication to docker's hub)
               subprocess.run("docker run --rm -v $PWD/logs:/host -v ai_system:/ai_system -w /ai_system busybox cp /ai_system/" + sender+"/knowledgeBase/currentSolution.h5 /host/"+sender+"_currentSolution.h5", shell=True)
               # 3. build ANN-based container for relevant architectures in current build context folder and publish at docker's hub
               subprocess.run("docker buildx build --platform linux/arm/v7,linux/arm64/v8,linux/amd64 --file "+logDirectory+"/"+sender+"-docker-file --tag marcusgrum/knowledgebase_"+sender+":latest --push  "+logDirectory+"/", shell=True)

          if (sub_process_method == "parallel"):
               # 2. copy ANN to dockers current build context folder (for preparing publication to docker's hub)
               with open(logDirectory+"/"+sender+"_stdout.txt", "wb") as out, open(logDirectory+"/"+sender+"_stderr.txt", "wb") as err:
                    subprocess.Popen("docker run --rm -v $PWD/logs:/host -v ai_system:/ai_system -w /ai_system busybox cp /ai_system/"+sender+"/knowledgeBase/currentSolution.h5 /host/"+sender+"_currentSolution.h5", shell=True, stdout=out, stderr=err)
               # 3. build ANN-based container for relevant architectures in current build context folder and publish at docker's hub
               with open(logDirectory+"/"+sender+"_stdout.txt", "wb") as out, open(logDirectory+"/"+sender+"_stderr.txt", "wb") as err:
                    subprocess.Popen("docker buildx build --platform linux/arm/v7,linux/arm64/v8,linux/amd64 --file "+logDirectory+"/"+sender+"-docker-file --tag marcusgrum/knowledgebase_"+sender+":latest --push  "+logDirectory+"/", shell=True, stdout=out, stderr=err)

     if (scenario == 'realize_annExperiment'):
          # comment out to keep current results and avoid accidental activation (unintended overwriting containers)
          #experiment01.realize_experiment()
          #experiment02.realize_experiment()
          #experiment03.realize_experiment()
          #experiment04.realize_experiment()
          #experiment05.realize_experiment()
          pass
     
     # for experiment05
     if (scenario == 'apply_annSolution_for_imageClassification') or (scenario == 'apply_annSolution_for_transportClassification') or (scenario == 'manual_sensorValueUpdate'):
          if (sub_process_method == "parallel"):
               # b) by subprocess.Popen()
               # Remark: By this variant, parallel requests at the same machine are realized in parallel. Hence, individual stdout and stderr have been created so that CLI output is separated correctly.
               # Please note, message broaker does not manage requests. Indeed, each machine requires a manager for efficient ressource allocation.
               with open(logDirectory+"/"+sender+"_stdout.txt", "wb") as out, open(logDirectory+"/"+sender+"_stderr.txt", "wb") as err:
                    # carry out current scenario
                    p = subprocess.Popen("docker-compose -f "+logDirectory+"/"+sender+"-docker-compose.yml up --remove-orphans", shell=True, stdout=out, stderr=err)
                    print('Message of ' + sender + ' has been triggered at ' + receiver + ' successfully!')
                    # wait for finalization at experiment 5
                    p.wait(timeout=None)
                    if (scenario == 'apply_annSolution_for_imageClassification'):
                         # announce finalization of ANN requests
                         client.publish(MQTT_Topic_CoNM, 'This is a result indication! My name is '+ hostName+' and I have processed the ann request.')
                    if (scenario == 'apply_annSolution_for_transportClassification'):
                         # announce finalization of ANN requests
                         client.publish(MQTT_Topic_CoNM, 'This is a result indication! My name is '+ hostName+' and I have processed the ann request.')
                    if (scenario == 'manual_sensorValueUpdate'): # this can be used for testing
                         # announce finalization of sensor update
                         # maybe put this in a separate client, so that ANN requests can be realized at AI-Lab 
                         # and decentralized systems can be virtually realized at production systems...?
                         client.publish(MQTT_Topic_CoNM, 'This is a sonsor update indication! My name is '+ hostName+' and I have updated the files by given values.')


if __name__ == '__main__':
     """
     This function initiates communication client
     and manages the corresponding AI reguests.
     Optional ToDo: 
     - Pull images for having most recent updates. 
     - Current code assumes images to be static (not changing over time).
     - If continual changes occure, a new AI case (and corresponding images) are released.
     """

     # optionally input parameters from CLI to rename host
     if (sys.argv[1] != ""):
          hostName = sys.argv[1]

     # specify client for messaging
     client = mqtt.Client()
     client.on_connect = on_connect
     client.on_message = on_message
     client.username_pw_set(username="user1", password="password1")

     # specify server for messaging
     global MQTT_Broker
     # MQTT_Broker = "test.mosquitto.org" # world wide network via public test server (communication can be seen by everyone)
     # MQTT_Broker = "broker.hivemq.com" # world wide network via public test server (communication can be seen by everyone)
     # MQTT_Broker = "iot.eclipse.org"   # world wide network via public test server (communication can be seen by everyone)
     # communication in local network (start server with /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf )
     MQTT_Broker = "localhost"

     # establish connection of client and server
     # - Method 1 - connect via plain MQTT protocol
     client.connect(MQTT_Broker, 1883, 60)
     # - Method 2 - connect via secure MQTT over TLS/SSL
     # TBD when required
     # - Method 3 - connect via MQTT over TLS/SSL with certificates
     # TBD when required
     # - Method 4 - connect via plain WebSockets configuration
     # TBD when required
     # - Method 5 - connect via WebSockets over TLS/SSL
     # TBD when required

     # Blocking call that processes network traffic, dispatches callbacks and
     # handles reconnecting.
     # Other loop*() functions are available that give a threaded interface and a
     # manual interface.

     # specify topics for subscriptions
     MQTT_Topic_CoNM = 'CoNM/workflow_system'
     # ...

     # optionally announce presence of client at server's topic-specific message channel
     client.publish(MQTT_Topic_CoNM, 'Hi there! My name is '+ hostName+' and I have subscribed to topic '+ MQTT_Topic_CoNM+'.')
     # ...

     # start listening here
     client.loop_forever()