""" 
This application realizes experimentation on artificial neural networks (ANN)
being used to coordinate multiple ANN-instructed systems
of different (a) ANN activation rates as well as (b) activation cycles.
For this, it simulates to kinds of systems (1) cps1 and (2) cps2.
Each cps realizes to kinds of sequential ANN-based tasks and so instruct their individual actuators:
(I) An ANN-based image classification. 
Here, it loads an ANN-based solution (pickle-file / xml-file) from the 'knowledgeBase' of docker volume 'ai_system' 
and activates it with image classification outcomes from the 'activationBase' of docker volume 'ai_system'.
Then, activation results are stored at the 'activationBase' of docker volume 'ai_system'.
For this, a TensorFlow-based library is used from the 'codeBase' of docker volume 'ai_system'.
(II) An ANN-based transport classification.
Here, it loads an ANN-based solution (PyBrain-file / xml-file) from the 'knowledgeBase' of docker volume 'ai_system' 
and activates it with image classification outcomes from 'activationBase' of docker volume 'ai_system',
which are produced by the previous ANN task (I), and current cps's sensor values.
Then, activation results are stored at the 'activationBase' of docker volume 'ai_system'.
For this, a custom PyBrain-based library is used from the 'codeBase' of docker volume 'ai_system'.

    Copyright (C) 2024>  Dr.-Ing. Marcus Grum

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
    
    TBD when required:
    - include all realize_experiment_id_x() from experiment05_v04_correctTimings.py
    - modify all realize_experiment_id_x() so that all tasks are started (nearly) simultaneously / parallelly that provide the same activation time start
"""

__author__ = 'Marcus Grum, marcus.grum@uni-potsdam.de'
# SPDX-License-Identifier: AGPL-3.0-or-later or individual license
# SPDX-FileCopyrightText: 2022 Marcus Grum <marcus.grum@uni-potsdam.de>

# library imports
import time
import simpy
import paho.mqtt.publish as publish
import paho.mqtt.client as mqtt
import json
import os
import platform

# Specify global variables, so that they are known 
# (1) at messageClient start and (2) at function calls from external scripts
global hostName, hostArch, logDirectory
hostName = os.uname()[1]
hostArch = platform.machine()
logDirectory = "./logs"  # = $PWD/logs

class CPS(object):

    global standardCycleLength, standardTimeFormat, client2, MQTT_Topic_CoNM

    standardCycleLength = 60
    standardTimeFormat = "%M:%S" # "%H:%M:%S"

    # The callback for when the client2 receives a CONNACK response from the server.
    def on_connect(client2, userdata, flags, rc):
        """
        This function subscribse client in on_connect().
        Subscribing in on_connect() means that if we lose the connection and reconnect, then subscriptions will be renewed.
        """

        print("Instance messaging: Connected with result code "+str(rc))
        client2.subscribe(MQTT_Topic_CoNM, qos=0)  # channel to deal with CoNM

    # The callback for when a PUBLISH message is received from the server.
    def on_message(client2, userdata, msg):
        """
        This function continuously receives messages from broker and starts scenario realization.
        It can be called via the following CLI commands:
        1. Initiate example apply_annSolution from remote:
        mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=apply_annSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_20, activation_base=marcusgrum/activationbase_apple_okay_01, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
        2. Initiate example create_annSolution from remote:
        mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=create_annSolution, knowledge_base=-, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
        3. Initiate example refine_annSolution from remote:
        mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=refine_annSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_01, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
        4. Initiate example wire_annSolution from remote:
        mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=wire_annSolution, knowledge_base=-, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
        5. Initiate example publish_annSolution from remote:
        mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=publish_annSolution, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
        6. Initiate experiment realize_annExperiment from remote:
        mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=realize_annExperiment, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
        """

        verbose = False

        # unroll messages
        message = msg.payload.decode()
        topic = msg.topic
        if verbose: print('       Instance messaging: '+msg.topic + " " + str(message))
        scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver = unroll_message(str(message))

        # or realize current scenario remotely
        global waitingForCurrentAnnReply_systemClock, waitingForCurrentAnnReply_cps1, waitingForCurrentAnnReply_cps2
        if('result indication' in message):
            if('cps1' in message):
                waitingForCurrentAnnReply_cps1 = False
                if not verbose: print('       Instance messaging: '+msg.topic + " " + str(message))
            elif('cps2' in message):
                waitingForCurrentAnnReply_cps2 = False
                if not verbose: print('       Instance messaging: '+msg.topic + " " + str(message))
            if verbose: print('       Instance messaging: Message of ' + sender + ' has been initiated at ' + receiver + ' by ' + hostName + ' successfully!')
    
    # Specify client2 for messaging.
    client2 = mqtt.Client()
    client2.on_connect = on_connect
    client2.on_message = on_message
    client2.username_pw_set(username="user1", password="password1")

    # Specify server for messaging.
    global MQTT_Broker
    # MQTT_Broker = "test.mosquitto.org" # world wide network via public test server (communication can be seen by everyone)
    # MQTT_Broker = "broker.hivemq.com" # world wide network via public test server (communication can be seen by everyone)
    # MQTT_Broker = "iot.eclipse.org"   # world wide network via public test server (communication can be seen by everyone)
    # communication in local network (start server with /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf )
    MQTT_Broker = "localhost"

    # Establish connection of client2 and server.
    # - Method 1 - connect via plain MQTT protocol
    client2.connect(MQTT_Broker, 1883, 60)
    # - Method 2 - connect via secure MQTT over TLS/SSL
    # TBD when required
    # - Method 3 - connect via MQTT over TLS/SSL with certificates
    # TBD when required
    # - Method 4 - connect via plain WebSockets configuration
    # TBD when required
    # - Method 5 - connect via WebSockets over TLS/SSL
    # TBD when required

    # Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a manual interface.

    # Specify topics for subscriptions.
    MQTT_Topic_CoNM = 'CoNM/workflow_system'

    # Optionally announce presence of client2 at server's topic-specific message channel.
    client2.publish(MQTT_Topic_CoNM, 'Instance messaging: Hi there! My name is ' + hostName +', I am caring about simulation controle and I have subscribed to topic '+ MQTT_Topic_CoNM+'.')
    
    # Start a non-blocking thread to wait for messages.
    client2.loop_start()

    global waitingForCurrentAnnReply_systemClock, waitingForCurrentAnnReply_cps1, waitingForCurrentAnnReply_cps2
    waitingForCurrentAnnReply_systemClock = False
    waitingForCurrentAnnReply_cps1 = False
    waitingForCurrentAnnReply_cps2 = False

    def __init__(self, env, name, activationRate, activationDelay):
        
        self.env = env
        self.name = name
        self. activationRate = activationRate
        self.activationDelay = activationDelay

        self.client2 = mqtt.Client()
        self.client2.on_connect = self.on_connect
        #self.client2.on_publish = self.on_publish

        # Optionally announce presence of client2 at server's topic-specific message channel.
#        client2.publish(MQTT_Topic_CoNM, 'Instance messaging: Hi there! My name is '+ self.name +' and I have subscribed to topic '+ MQTT_Topic_CoNM+'.')

        # Start the run process everytime an instance is created.
        self.action = env.process(self.run())


    def run(self):

        global waitingForCurrentSensorUpdateReply_cps1, waitingForCurrentSensorUpdateReply_cps2
        global waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2
        global waitingForCurrentAnnReply_systemClock, waitingForCurrentAnnReply_cps1, waitingForCurrentAnnReply_cps2

        verbose = False

        yield self.env.process(self.delay_first_activationStart(self.activationDelay))

        while True:

            # Indicate start of work.
            print(self.name + ': Start working at ' + str(time.strftime(standardTimeFormat, time.gmtime(self.env.now))))

            # Realize ANN-based analyses for cps1 and cps2.
            if ('system_clock' not in self.name):
                
                # Step 1
                print('   ' + self.name + ' is requesting for ANN-based picture analysis (instance printing).')
                client2.publish(MQTT_Topic_CoNM, '   Instance messaging: Hi there! My name is the simulated instance of '+ self.name +' and I need the real instance of mine to do an ANN-based picture analysis.')

                # Wait for current ANN analysis result of step 1.
                if ('cps1' in self.name):
                    waitingForCurrentSensorUpdateReply_cps1 = True
                    waitingForAnnReplies_cps1 = True
                    waitingForCurrentAnnReply_cps1 = True
                elif ('cps2' in self.name):
                    waitingForCurrentSensorUpdateReply_cps2 = True
                    waitingForAnnReplies_cps2 = True
                    waitingForCurrentAnnReply_cps2 = True
                if verbose: print('       Instance current states: ', waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2, waitingForCurrentAnnReply_systemClock, waitingForCurrentAnnReply_cps1, waitingForCurrentAnnReply_cps2)
                i = 0
                while (waitingForCurrentAnnReply_systemClock or waitingForCurrentAnnReply_cps1 or waitingForCurrentAnnReply_cps2):
                    if verbose: print ('       Instance waiting cue:', i, waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2, waitingForCurrentAnnReply_systemClock, waitingForCurrentAnnReply_cps1, waitingForCurrentAnnReply_cps2)
                    time.sleep(1)
                    i = i + 1
                
                # Step 2
                print('   ' + self.name + ' is requesting for ANN-based transport (instance printing).')
                client2.publish(MQTT_Topic_CoNM, '   Instance messaging: Hi there! I am the simulated instance of '+ self.name +' and I need the real instance of mine to do an ANN-based transport analysis.')

                # Wait for current ANN analysis result of step 2.
                if ('cps1' in self.name):
                    #waitingForCurrentSensorUpdateReply_cps1 = True
                    #waitingForAnnReplies_cps1 = True
                    waitingForCurrentAnnReply_cps1 = True
                elif ('cps2' in self.name):
                    #waitingForCurrentSensorUpdateReply_cps2 = True
                    #waitingForAnnReplies_cps2 = True
                    waitingForCurrentAnnReply_cps2 = True
                if verbose: print('       Instance current states: ', waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2, waitingForCurrentAnnReply_systemClock, waitingForCurrentAnnReply_cps1, waitingForCurrentAnnReply_cps2)
                i = 0
                while (waitingForCurrentAnnReply_systemClock or waitingForCurrentAnnReply_cps1 or waitingForCurrentAnnReply_cps2):
                    if verbose: print ('       Instance waiting cue:', i, waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2, waitingForCurrentAnnReply_systemClock, waitingForCurrentAnnReply_cps1, waitingForCurrentAnnReply_cps2, waitingForCurrentSensorUpdateReply_cps1, waitingForCurrentSensorUpdateReply_cps2)
                    time.sleep(1)
                    i = i + 1
                    
                # Indicate finalization of ANNs at this time step for simulation control.
                client2.publish(MQTT_Topic_CoNM, '   Instance messaging: The results of this timestep have been produced at '+ self.name +'.')                

            # Progress simulated activity of current simulation time step.
            working_duration = standardCycleLength * self.activationRate
            yield self.env.process(self.work(working_duration))
            if verbose: print('             Instance messaging: end of current run cycle has been reached.')

    def delay_first_activationStart(self, duration):
        """
        This function deleays start of simulated system by consuming its simulation time.
        """

        yield self.env.timeout(duration)

    def work(self, duration):
        """
        This function carries out work of simulated system by consuming its simulation time.
        """

        yield self.env.timeout(duration)

# set up client to wait for messages
def on_connect(client, userdata, flags, rc):
    """
    This function subscribse client in on_connect().
    Subscribing in on_connect() means that if we lose the connection and reconnect, then subscriptions will be renewed.
    """

    print("Global messaging: Connected with result code "+str(rc))
    client.subscribe(MQTT_Topic_CoNM, qos=0)  # channel to deal with CoNM

def unroll_message(message):
    """
    This function unrolls variables from message and returns them.
    """

    scenario = (message.partition("scenario=")[2]).partition(", knowledge_base=")[0]
    knowledge_base = (message.partition("knowledge_base=")[2]).partition(", activation_base=")[0]
    activation_base = (message.partition("activation_base=")[2]).partition(", code_base=")[0]
    code_base = (message.partition("code_base=")[2]).partition(", learning_base=")[0]
    learning_base = (message.partition("learning_base=")[2]).partition(", sender=")[0]
    sender = (message.partition("sender=")[2]).partition(", receiver=")[0]
    receiver = (message.partition("receiver=")[2]).partition(".")[0]

    return scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver

def on_message(client, userdata, msg):
    """
    This function continuously receives messages from broker and starts scenario realization.
    It can be called via the following CLI commands:
    1. Initiate example apply_annSolution from remote:
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=apply_annSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_20, activation_base=marcusgrum/activationbase_apple_okay_01, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    2. Initiate example create_annSolution from remote:
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=create_annSolution, knowledge_base=-, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    3. Initiate example refine_annSolution from remote:
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=refine_annSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_01, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    4. Initiate example wire_annSolution from remote:
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=wire_annSolution, knowledge_base=-, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    5. Initiate example publish_annSolution from remote:
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=publish_annSolution, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    6. Initiate experiment realize_annExperiment from remote:
    mosquitto_pub -t "CoNM/workflow_system" -u user1 -P password1 -m "Please realize the following AI case: scenario=realize_annExperiment, knowledge_base=-, activation_base=-, code_base=-, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    """

    verbose = False

    # Provide variables as global so that these are known in this thread.
    # global hostName
    global waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2
    global waitingForCurrentSensorUpdateReply_cps1, waitingForCurrentSensorUpdateReply_cps2

    # Unroll messages.
    message = msg.payload.decode()
    topic = msg.topic
    if verbose: print('       Global messaging: '+msg.topic + " " + str(message))
    scenario, knowledge_base, activation_base, code_base, learning_base, sender, receiver = unroll_message(str(message))

    # Guarantee work finalization of corresponding systems.
    if('The results of this timestep have been produced' in message):
        if('cps1' in message):
            waitingForAnnReplies_cps1 = False
            if not verbose: print('       Global messaging via topic ('+msg.topic + "): " + str(message))
        elif('cps2' in message):
            waitingForAnnReplies_cps2 = False
            if not verbose: print('       Global messaging via topic ('+msg.topic + "): " + str(message))
    if('This is a sonsor update indication' in message):
        if('cps1' in message):
            waitingForCurrentSensorUpdateReply_cps1 = False
            if not verbose: print('       Global messaging via topic ('+msg.topic + "): " + str(message))
        elif('cps2' in message):
            waitingForCurrentSensorUpdateReply_cps2 = False
            if not verbose: print('       Global messaging via topic ('+msg.topic + "): " + str(message))
    
    # Show current simulation controle activity
    if verbose: print('       Global messaging: Message of ' + sender + ' for ' + receiver + ' has been received at simulation control by ' + hostName + ' successfully!')

def carry_out_simulation(env, until):

    # Specify global variables.
    global waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2
    global waitingForCurrentSensorUpdateReply_cps1, waitingForCurrentSensorUpdateReply_cps2
    verbose = False

    current_time = 0
    while env.peek() < until:

        # Having a look to next simulation step to identify all activities per current_time
        if verbose: print('env.peek()=', env.peek())

        # Carry out ANN activation of current_time.
        if(current_time == env.peek()):
            env.step()
            time.sleep(1)
        # If all ANN activations have been identified at current_time,
        # carry out ANN-based instructions.
        else:
            i = 0
            # Indeed, this loop is not activated since currently active cps instance and its message client 
            # is blocking this simulation controle and its message client.
            # Finally, the last activation of currently active cps releases the waitingForAnnReplies variables,
            # so that this while loop will be activated by non-SimPy systems.
            while (waitingForAnnReplies_systemClock or waitingForAnnReplies_cps1 or waitingForAnnReplies_cps2 or waitingForCurrentSensorUpdateReply_cps1 or waitingForCurrentSensorUpdateReply_cps2):
                if verbose: print ('       Global waiting cue:', i, waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2, waitingForCurrentSensorUpdateReply_cps1, waitingForCurrentSensorUpdateReply_cps2)
                time.sleep(1)
                i = i + 1
            
            # Realize physical transportation of cps1 and cps2 and sensor value updates
            # ...realized decentrally at individual physical cps!
            
            # Update current_time so that next simulated timestep can be identified.
            # So, for instance, real-time simulations (activating all cps at one time step) can be realized when using non-blocking SimPy instances.
            if verbose: print('updating current_time with ', env.peek())
            current_time = env.peek()

def realize_experiment_id_01():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (on the beat)
    """

    # Specify simulation.
    print("Realizing experiment ID01...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=1, activationDelay=0)

    # Carry out simulation.
    until = 180
    carry_out_simulation(env, until)

def realize_experiment_id_02():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (on the offbeat)
    """

    # Specify simulation.
    print("Realizing experiment ID02...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=1, activationDelay=15)

    # Carry out simulation.
    until = 180
    carry_out_simulation(env, until)

def realize_experiment_id_03():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (on the interval-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID03...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=1/3, activationDelay=0)

    # Carry out simulation.
    until = 180
    carry_out_simulation(env, until)

def realize_experiment_id_04():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (on the manifold-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID04...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=3, activationDelay=0)

    # Carry out simulation.
    until = 180*3
    carry_out_simulation(env, until)

def realize_experiment_id_05():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (of the interval-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID05...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=1/numpy.pi, activationDelay=0)

    # Carry out simulation.
    until = 180
    carry_out_simulation(env, until)

def realize_experiment_id_06():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (off the manifold-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID06...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=1*numpy.pi, activationDelay=0)

    # Carry out simulation.
    until = 180*numpy.pi
    carry_out_simulation(env, until)

def realize_experiment_id_07():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    chaotic (off the beat)
    """

    # Specify simulation.
    print("Realizing experiment ID07...")
    env = simpy.Environment()
    currentVarRateCps2 = 1/(60/19)
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=currentVarRateCps2, activationDelay=0)
    # Carry out simulation.
    until = 60 * currentVarRateCps2
    carry_out_simulation(env, until)

    # Specify modified simulation.
    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    # Specify modified simulation.
    currentVarRateCps2 = 1/(60/26)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    # Specify modified simulation.
    currentVarRateCps2 = 1/(60/18)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation
    carry_out_simulation(env, until)

    # Specify modified simulation.
    currentVarRateCps2 = 1/(60/24)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    # Specify modified simulation.
    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    # Specify modified simulation.
    currentVarRateCps2 = 1/(60/88)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

def realize_experiment_id_08():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    chaotic (no beat)
    """

    # Specify simulation.
    print("Realizing experiment ID08...")
    env = simpy.Environment()

    # initiating cycle 1
    currentVarRateCps1 = 1/(60/48)
    currentVarRateCps2 = 1/(60/19)
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=currentVarRateCps1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=currentVarRateCps2, activationDelay=0)
    until = 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    cps1.activationRate = 1/(60/58)
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/26)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    # initiating cycle 2
    currentVarRateCps2 = 1/(60/18)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/24)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    cps1.activationRate = 1/(60/50)
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/88)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

def realize_experiment_id_09():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (on shifted interval-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID09...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=1/3, activationDelay=15)

    # Carry out simulation.  
    until = 180
    carry_out_simulation(env, until)

def realize_experiment_id_10():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (on shifted manifold-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID10...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=15)
    cps2 = CPS(env, name='cps2', activationRate=3, activationDelay=0)

    # Carry out simulation
    until = 180*3
    carry_out_simulation(env, until)

def realize_experiment_id_11():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (off shifted interval-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID11...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=1/numpy.pi, activationDelay=2)

    # Carry out simulation.
    until = 180
    carry_out_simulation(env, until)

def realize_experiment_id_12():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    periodically (off shifted manifold-beat)
    """

    # Specify simulation.
    print("Realizing experiment ID12...")
    env = simpy.Environment()
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=2)
    cps2 = CPS(env, name='cps2', activationRate=1*numpy.pi, activationDelay=0)

    # Carry out simulation.
    until = 180*numpy.pi
    carry_out_simulation(env, until)

def realize_experiment_id_13():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    chaotic (off the shifted beat)
    """

    # Specify simulation.
    print("Realizing experiment ID13...")
    env = simpy.Environment()

    currentVarRateCps2 = 1/(60/19)
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=currentVarRateCps2, activationDelay=2)
    until = 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/26)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/18)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/24)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/88)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

def realize_experiment_id_14():
    """
    This function carries out simulation of ANN-instructed cps1 and ANN-instructed cps2
    providing activation rates and activation cycles of the following system state characteristics:
    chaotic (shifted no beat)
    """

    # Specify simulation.
    print("Realizing experiment ID14...")
    env = simpy.Environment()

    # initiating cycle 1
    currentVarRateCps1 = 1/(60/48)
    currentVarRateCps2 = 1/(60/19)
    systemclock = CPS(env, name='system_clock', activationRate=1, activationDelay=0)
    cps1 = CPS(env, name='cps1', activationRate=currentVarRateCps1, activationDelay=0)
    cps2 = CPS(env, name='cps2', activationRate=currentVarRateCps2, activationDelay=2)
    until = 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    cps1.activationRate = 1/(60/58)
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/26)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    # initiating cycle 2
    currentVarRateCps2 = 1/(60/18)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/24)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/9)
    cps2.activationRate = currentVarRateCps2
    cps1.activationRate = 1/(60/50)
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

    currentVarRateCps2 = 1/(60/88)
    cps2.activationRate = currentVarRateCps2
    until = until + 60 * currentVarRateCps2
    # Carry out simulation.
    carry_out_simulation(env, until)

def realize_experiment():

    # specify client for messaging
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.username_pw_set(username="user1", password="password1")

    # Specify server for messaging.
    global MQTT_Broker
    # MQTT_Broker = "test.mosquitto.org" # world wide network via public test server (communication can be seen by everyone)
    # MQTT_Broker = "broker.hivemq.com" # world wide network via public test server (communication can be seen by everyone)
    # MQTT_Broker = "iot.eclipse.org"   # world wide network via public test server (communication can be seen by everyone)
    # communication in local network (start server with /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf )
    MQTT_Broker = "localhost"

    # Establish connection of client and server.
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

    # Blocking call that processes network traffic, dispatches callbacks and handles reconnecting.
    # Other loop*() functions are available that give a threaded interface and a manual interface.

    # Specify topics for subscriptions.
    # So far, only one topic has been used for simple message printing.
    MQTT_Topic_CoNM = 'CoNM/workflow_system'

    # Optionally announce presence of client at server's topic-specific message channel.
    client.publish(MQTT_Topic_CoNM, 'Global messaging: Hi there! My name is ' + hostName +', I am caring about simulation controle and I have subscribed to topic '+ MQTT_Topic_CoNM+'.')

    # Start a non-blocking thread to wait for messages.
    client.loop_start()

    # Realize non-blocking experiment realization that allows receiving of managing mqtt messages.
    global waitingForAnnReplies_systemClock, waitingForAnnReplies_cps1, waitingForAnnReplies_cps2
    global waitingForCurrentSensorUpdateReply_cps1, waitingForCurrentSensorUpdateReply_cps2
    waitingForAnnReplies_systemClock = False
    waitingForAnnReplies_cps1 = False
    waitingForAnnReplies_cps2 = False
    waitingForCurrentSensorUpdateReply_cps1 = False
    waitingForCurrentSensorUpdateReply_cps2 = False

    # Start selected experiments.
    # Comment out to keep current results and avoid accidental activation
    realize_experiment_id_01()
    #realize_experiment_id_02()
    #realize_experiment_id_03()
    ##realize_experiment_id_04()
    ##realize_experiment_id_05()
    ##realize_experiment_id_06()
    ##realize_experiment_id_07()
    ##realize_experiment_id_08()
    #realize_experiment_id_09()
    ##realize_experiment_id_10()
    ##realize_experiment_id_11()
    ##realize_experiment_id_12()
    ##realize_experiment_id_13()
    #realize_experiment_id_14()

    # Don't leave a zombie thread behind.
    client.loop_stop()

if __name__ == '__main__':
    
    # when script is started manually, initiate experiment incl. plotting
    realize_experiment()

    # comment out to keep current results and avoid accidental activation
    pass