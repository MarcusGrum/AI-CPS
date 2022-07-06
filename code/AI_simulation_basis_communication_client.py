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

def load_data_from_CsvFile(path):
    """
    This functions loads csv data from the 'path' and returns it.
    """ 
    data = []
    with open(path + '.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='"') # alternative delimiters '\t', ';', alternative quotechars '"', '|'
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
    client.subscribe(MQTT_Topic_CoNM, qos=0) # channel to deal with CoNM
    # ...

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    
    # provide variables as global so that these are known in this thread
    #global hostName
    
    # unroll messages
    message = msg.payload.decode()
    topic = msg.topic
    print(msg.topic + " " + str(message))
    scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver = unroll_message(str(message))
    
    # build docker-compose file based on message
    if(scenario == 'apply_knnSolution'):
        build_docker_compose_file_for_apply_knnSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
    if(scenario == 'create_knnSolution'):
        build_docker_compose_file_for_create_knnSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)
    if(scenario == 'refine_knnSolution'):
        build_docker_compose_file_for_refine_knnSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver)

    # realize instructions from messages
    # by running docker-compose file created at current working directory
    subprocess.call("docker-compose -f docker-compose.yml up", shell=True)
    print('Message of ' + sender + ' has been processed at ' + receiver + ' successfully!')

def unroll_message(message):
    """
    This functions unrolls variables from message and returns them.
    """ 

    scenario        = (message.partition("scenario=")[2]).partition(", knowledge_base=")[0]
    knowledge_base  = (message.partition("knowledge_base=")[2]).partition(", activation_base=")[0]
    activation_base = (message.partition("activation_base=")[2]).partition(", code_base=")[0]
    code_base       = (message.partition("code_base=")[2]).partition(", learning_base=")[0]
    learning_base   = (message.partition("learning_base=")[2]).partition(", sender=")[0]
    sender          = (message.partition("sender=")[2]).partition(", receiver=")[0]
    receiver        = (message.partition("receiver=")[2]).partition(".")[0]

    return scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver

def build_docker_compose_file_for_apply_knnSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
    """
    This functions builds docker-compose file for scenario called apply_knnSolution
    and considers variables from message, here.
    The file is stored at current working directory.
    """ 

    # if architecture = 'x86_64'
    if(hostArch=='x86_64'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "3.0"'+'\n')
            f.write('services:'+'\n')
            f.write('  knowledge_base:'+'\n')
            f.write('    image: ' + knowledge_base + ''+'\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_20
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/;'+'\n')
            f.write('  activation_base:'+'\n')
            f.write('    image: ' + activation_base + ''+'\n') # e.g. marcusgrum/activationbase_apple_okay_01
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/activationBase/ && cp -r /activationBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "knowledge_base"'+'\n')
            f.write('      - "activation_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/apply_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

    # if architecture = 'x86_64_gpu'
    if(hostArch=='x86_64_gpu'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
            f.write('services:'+'\n')
            f.write('  knowledge_base:'+'\n')
            f.write('    image: ' + knowledge_base + ''+'\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_20
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/;'+'\n')
            f.write('  activation_base:'+'\n')
            f.write('    image: ' + activation_base + ''+'\n') # e.g. marcusgrum/activationbase_apple_okay_01
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/activationBase/ && cp -r /activationBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
            f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
            f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
            f.write('    # daemon.json.'+'\n')
            f.write('    runtime: nvidia'+'\n')
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "knowledge_base"'+'\n')
            f.write('      - "activation_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/apply_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

    # if architecture = 'aarch64'
    if(hostArch=='aarch64'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "3.9"'+'\n')
            f.write('services:'+'\n')
            f.write('  knowledge_base:'+'\n')
            f.write('    image: ' + knowledge_base + ''+'\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_20
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/;'+'\n')
            f.write('  activation_base:'+'\n')
            f.write('    image: ' + activation_base + ''+'\n') # e.g. marcusgrum/activationbase_apple_okay_01
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/activationBase/ && cp -r /activationBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    user: root'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "knowledge_base"'+'\n')
            f.write('      - "activation_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/apply_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

def build_docker_compose_file_for_create_knnSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
    """
    This functions builds docker-compose file for scenario called create_knnSolution
    and considers variables from message, here.
    The file is stored at current working directory.
    """ 

    # if architecture = 'x86_64'
    if(hostArch=='x86_64'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "3.0"'+'\n')
            f.write('services:'+'\n')
            f.write('  learning_base:'+'\n')
            f.write('    image: ' + learning_base + ''+'\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/learningBase/ && cp -r /learningBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "learning_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/create_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

    # if architecture = 'x86_64_gpu'
    if(hostArch=='x86_64_gpu'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
            f.write('services:'+'\n')
            f.write('  learning_base:'+'\n')
            f.write('    image: ' + learning_base + ''+'\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/learningBase/ && cp -r /learningBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
            f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
            f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
            f.write('    # daemon.json.'+'\n')
            f.write('    runtime: nvidia'+'\n')
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "learning_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/create_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

    # if architecture = 'aarch64'
    if(hostArch=='aarch64'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "3.9"'+'\n')
            f.write('services:'+'\n')
            f.write('  learning_base:'+'\n')
            f.write('    image: ' + learning_base + ''+'\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/learningBase/ && cp -r /learningBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    user: root'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "learning_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/create_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

def build_docker_compose_file_for_refine_knnSolution(scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver):
    """
    This functions builds docker-compose file for scenario called refine_knnSolution
    and considers variables from message, here.
    The file is stored at current working directory.
    """ 

    # if architecture = 'x86_64'
    if(hostArch=='x86_64'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "3.0"'+'\n')
            f.write('services:'+'\n')
            f.write('  learning_base:'+'\n')
            f.write('    image: ' + learning_base + ''+'\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/learningBase/ && cp -r /learningBase/ /tmp/;'+'\n')
            f.write('  knowledge_base:'+'\n')
            f.write('    image: ' + knowledge_base + ''+'\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "learning_base"'+'\n')
            f.write('      - "knowledge_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/refine_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

    # if architecture = 'x86_64_gpu'
    if(hostArch=='x86_64_gpu'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "2.3"  # the only version where "runtime" option is supported'+'\n')
            f.write('services:'+'\n')
            f.write('  learning_base:'+'\n')
            f.write('    image: ' + learning_base + ''+'\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/learningBase/ && cp -r /learningBase/ /tmp/;'+'\n') 
            f.write('  knowledge_base:'+'\n')
            f.write('    image: ' + knowledge_base + ''+'\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_x86_64_gpu !!!!
            f.write('    # Make Docker create the container with NVIDIA Container Toolkit'+'\n')
            f.write('    # You do not need it if you set nvidia as the default runtime in'+'\n')
            f.write('    # daemon.json.'+'\n')
            f.write('    runtime: nvidia'+'\n')
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "learning_base"'+'\n')
            f.write('      - "knowledge_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/refine_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

    # if architecture = 'aarch64'
    if(hostArch=='aarch64'):
        with open('./docker-compose.yml', 'w') as f:
            f.write('version: "3.9"'+'\n')
            f.write('services:'+'\n')
            f.write('  learning_base:'+'\n')
            f.write('    image: ' + learning_base + ''+'\n') # e.g. marcusgrum/learningbase_apple_banana_orange_pump_02
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/learningBase/ && cp -r /learningBase/ /tmp/;'+'\n')
            f.write('  knowledge_base:'+'\n')
            f.write('    image: ' + knowledge_base + ''+'\n') # e.g. marcusgrum/knowledgebase_apple_banana_orange_pump_01
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/knowledgeBase/ && cp -r /knowledgeBase/ /tmp/;'+'\n')
            f.write('  code_base:'+'\n')
            f.write('    user: root'+'\n')
            f.write('    image: ' + code_base + ''+'\n') # e.g. marcusgrum/codebase_ai_core_for_image_classification_aarch64
            f.write('    volumes:'+'\n')
            f.write('       - ai_system:/tmp'+'\n')
            f.write('    depends_on:'+'\n')
            f.write('      - "learning_base"'+'\n')
            f.write('      - "knowledge_base"'+'\n')
            f.write('    command:'+'\n')
            f.write('    - sh'+'\n')
            f.write('    - "-c"'+'\n')
            f.write('    - |'+'\n')
            f.write('      rm -rf /tmp/codeBase/ && cp -r /codeBase/ /tmp/;'+'\n')
            f.write('      python3 /tmp/codeBase/refine_knnSolution.py;'+'\n')
            f.write('volumes:'+'\n')
            f.write('  ai_system:'+'\n')
            f.write('    external: true'+'\n')

if __name__ == '__main__':
    """
    This function initiates communication client
    and manages the corresponding AI reguests.
    """
    global hostName, hostArch
    hostName = os.uname()[1]
    hostArch = platform.machine()
    if("AILab" in hostName):
        hostArch = hostArch + "_gpu"

    # specify client for messaging
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    #client.username_pw_set("username", "password")

    # specify server for messaging
    global MQTT_Broker
    MQTT_Broker = "test.mosquitto.org" # world wide network via public test server (communication can be seen by everyone)
    #MQTT_Broker = "broker.hivemq.com" # world wide network via public test server (communication can be seen by everyone)
    #MQTT_Broker = "iot.eclipse.org"   # world wide network via public test server (communication can be seen by everyone)
    #MQTT_Broker = "localhost"         # communication in local network

    # establish connection of client and server
    client.connect(MQTT_Broker, 1883, 60)

    # Blocking call that processes network traffic, dispatches callbacks and
    # handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a
    # manual interface.

    # specify topics for subscriptions
    MQTT_Topic_CoNM = 'CoNM/workflow_system'
    # ...

    # optionally announce presence of client at server's topic-specific message channel
    client.publish(MQTT_Topic_CoNM,'Hi there! My name is x and I have subscribed to topic ' + MQTT_Topic_CoNM + '.')
    # ...

    # start listening here
    client.loop_forever()