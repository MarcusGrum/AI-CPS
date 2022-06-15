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
    # ...
    
    # unroll messages
    message = msg.payload.decode()
    topic = msg.topic
    print(msg.topic + " " + str(message))
    
    if('mxy' in message):
        print('   do mxy!')
        subprocess.call("docker-compose -f ../../scenarios/apply_knnSolution/x86_64/docker-compose.yml up", shell=True)
    print('   message has been processed successfully!')

if __name__ == '__main__':
    """
    This function initiates communication client
    and manages the corresponding AI reguests.
    """

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
    client.publish(MQTT_Topic_CoNM,'Hi there! I am in and I have subscribed to topic ' + MQTT_Topic_CoNM + '.')
    # ...

    # start listening here
    client.loop_forever()