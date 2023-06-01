# Welcome to the Platform for AI-based Cyber-Physical Systems

The AI-CPS is an end-to-end open source platform for cyber-physical systems considering a knowledge base with the aid of artificial neuronal networks (ANN). 
It has a comprehensive, flexible ecosystem of tools, libraries, and community resources that lets researchers push the state-of-the-art in ANN and developers easily build and deploy ANN-powered applications. 
Its particular focus lays on the efficient application of knowledge of ANN.
For this, it enables the flexible, node-independent (a) situational `ANN application`, (b) `ANN training and validation` as well as (c) `ANN refinement`,
which is realized as Over-The-Air deployment of 
(1) ANN models considered as `knowledge base`,
(2) ANN activations considered as `activation base`,
(3) ANN training material considered as `learning base` and
(4) ANN routines considered as `codeBase`.
Among furthers, this is realized on behalf of `tensor_flow` libraries, `docker` mechanisms, `python` applications,
so that they can be realized on any kind of device (tested with `raspberry`, `mac`, `ubuntu`),
any kind of platform (tested with `aarch64`, `x86_64` and `x86_64_gpu`).

The tool was originally developed by Dr.-Ing. Marcus Grum.

## Getting Started

### 1) Set up operating system

#### 1.1) Set up OS at raspberry

##### Set up OS at raspberry via terminal

1. Download and install recent OS, e.g. `raspberry Pi OS Lite buster 12.02.2020`, on sd card via laptop.

1. Create ssh file an boot section.

1. Configure WLAN information.

1. Put SD card into raspberry and start device.

##### Alternatively, set up device via Raspberry Pi Imager

1. Download and install Raspberry Pi Imager (e.g. v1.7.2) from [Raspberry Pi OS Page](https://www.raspberrypi.com/software/).

1. Configure your device by selecting `raspberry Pi OS Lite (64bit)`, SD card and corresponding settings.

1. Hit `write os`.

1. Put SD card into raspberry and start device.

##### Log in to your raspberry

1. Change default password.

    ```
    sudo passwd pi
    ```

1. Connect on your raspberry via shell with user `pi`, e.g. with password `raspberry`:

    ```
    ssh pi@141.89.39.173
    ```

    or by

    ```
    ssh pi@AiLabraspberry1.local
    ```

1. Test your current distro by `lsb_release -a`.

1. Test your architecture by `uname -a`.

#### 1.2) Set up any kind of linux-based system

Please follow installation details of [AiLab Installation Guide](https://github.com/MarcusGrum/AI-Lab).
This shows the installation of numerous of the required tools.
Hence, the following just shows a raspberry focus and some extra commands.

### 2) Set up docker

#### 2.1) Set up docker at raspberry

This is based on the [Docker Installation Guide](https://dev.to/elalemanyo/how-to-install-docker-and-docker-compose-on-raspberry-pi-1mo).

1. Update your os and accept its 'Suite' value from 'testing' to 'oldstable' explicitly before updates for this repository can be applied.

    ```
    sudo apt-get update
    ```

1. Upgrade your os.

    ```
    sudo apt-get upgrade
    ```

1. Install docker

    ```
    curl -sSL https://get.docker.com | sh
    ```

1. Add a non-root user to the docker group.

    ```
    sudo usermod -aG docker ${USER}
    ```

1. Prepare installation of Docker-Compose.

    ```
    sudo apt-get install libffi-dev libssl-dev
    sudo apt install python3-dev
    sudo apt-get install -y python3 python3-pip
    ```

1. Install Docker-Compose.

    ```
    sudo pip3 install docker-compose
    ```

1. Enable the Docker system service to start your containers on boot.

    ```
    sudo systemctl enable docker
    ```

1. Restart your device.

    ```
    sudo reboot
    ```

1. Run Hello World Container for testing Docker installation.

    ```
    docker run hello-world
    ```

1. Create docker volume `ai_system` for using mechanisms of this repository:

	```
	docker volume create ai_system
	```

#### 2.2) Set up docker at mac

This is based on the [Docker Installation Guide](https://docs.docker.com/desktop/mac/install/).

1. Download recent `.dmg` file called `Docker Desktop for Mac` and install it.

1. Run Hello World Container for testing Docker installation.

    ```
    docker run hello-world
    ```

1. Create docker volume `ai_system` for using mechanisms of this repository:

	```
	docker volume create ai_system
	```

### 3) Set up tensorflow

#### 3.1) Set up tensorflow at raspberry
    
1. Find useful links, here:

    ```
    https://github.com/fgervais/docker-tensorflow/blob/master/Dockerfile
    https://github.com/armindocachada/raspberrypi-docker-tensorflow-opencv
    https://github.com/lhelontra/tensorflow-on-arm
    https://github.com/samjabrahams/tensorflow-on-raspberry-pi
    ```
    
1. Maybe include camera at pi.

    ```
    https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/
    ```

##### Use tensorflow container to run in an interactive bash at raspberry
    
1. Start container that provides all (SciPi, Matplotlib, etc.).

    ```
    docker run --rm -it armswdev/tensorflow-arm-neoverse bash
    ```   
    
    Alternatively, use light-weight containers.
    
    ```
    docker run --rm -it francoisgervais/tensorflow:2.1.0-cp35 bash
    ```
    
##### Use tensorflow container to run a small program at raspberry

1. Run tensorflow docker container and execute a small example program.

    ```
    docker run -it armswdev/tensorflow-arm-neoverse python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```

###### Use tensorflow container to run a script at raspberry

1. Run tensorflow docker container and execute a small example script.

    ```
    docker run -it \
    -v /home/pi/knowledgeBases:/home/ubuntu/knowledgeBases \
    armswdev/tensorflow-arm-neoverse  \
    python3 /home/ubuntu/knowledgeBases/testScript.py
    ```

###### Use tensorflow container to run in the background and initiate script execution at raspberry   
   
1. Run tensorflow docker container in background.

    ```
    docker run \
    -v /home/pi/knowledgeBases:/home/ubuntu/knowledgeBases \
    armswdev/tensorflow-arm-neoverse 
    ```

1. Get to know the container's name, which is in this example `37eb7158877c`.

    ```
    docker ps
    ```

1. Execute script.

    ```
    docker exec 37eb7158877c python3 /home/ubuntu/knowledgeBases/tensorflowWorkDir/testScript.py
    ``` 

#### 3.2) Set up tensorflow at mac

##### Use tensorflow container to run in an interactive bash at mac    
    
1. Start container.   
    
    ```
    docker pull tensorflow/tensorflow                     # latest stable release (cpu-based calculation), which is right now: 2.8.0
    ```
    
    Alternatively, use containers with an alternative configuration.
    
    ```
    docker pull tensorflow/tensorflow:latest-gpu            # "gpu" support; "latest" of the latest TensorFlow binary image, which is right now: 2.6.0
    docker pull tensorflow/tensorflow:version-gpu           # "gpu" support; "version" of the TensorFlow binary image, for example: 2.1.0
    ```
    
    For further information, have a look on [Access an NVIDIA GPU Information Section](https://docs.docker.com/engine/reference/commandline/run/#access-an-nvidia-gpu).
    
##### Use tensorflow container to run a small program at mac

1. Run tensorflow docker container and execute a small example program.

    ```
    docker run -it tensorflow/tensorflow python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```

##### Use tensorflow container to run a script at mac

1. Run tensorflow docker container and execute a small example script.

    ```
    docker run -it \
    -v /home/pi/knowledgeBases:/home/ubuntu/knowledgeBases \
    tensorflow/tensorflow  \
    python3 /home/ubuntu/knowledgeBases/testScript.py
    ```
    
    Alternatively, consider the tmp folder as current working directory
    
    ```
    docker run -it --rm \
    -v $PWD/repositories/tensorflowWorkDir:/tmp \
    -w /tmp \
    tensorflow/tensorflow \
    python3 ./testScript.py
    ```
    
#### 3.3) Build individual tensorflow containers and deploy them

1. Find useful links, here:

    ```
    https://github.com/tensorflow/build
    https://medium.com/tensorflow/tensorflow-1-9-officially-supports-the-raspberry-pi-b91669b0aa0
    https://www.tensorflow.org/install/docker
    https://github.com/tensorflow/build/tree/master/raspberry_pi_builds
    ```

1. Find individual instructions at corresponding readmes of this repository.
    
### 4) Set up MQTT

#### 4.1) Set up MQTT at raspberry

1. Install MQTT library:

   ```
   pip3 install paho-mqtt
   ```

#### 4.2) Set up MQTT at mac

1. Install MQTT library:

   ```
   pip3 install paho-mqtt
   ```

#### 4.3) Set up MQTT at linux

1. Install pip3 for installing MQTT library:

   ```
   sudo apt install python3-pip
   ```

1. Install MQTT library for communication client:

   ```
   pip3 install paho-mqtt
   ```

1. Install mosquitto server for MQTT broker:

	```
	sudo apt-get install mosquitto
	```

#### 4.4) Test MQTT installation

1. Start MQTT server at first CLI for managing communication:

	For Mac:
	
    ```
    /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf    
    ```

	For Ubuntu:
	
	```
	mosquitto -v
	```
	
	or
	
	```
	sudo /usr/sbin/mosquitto -c /etc/mosquitto/mosquitto.conf
	```	
	
	You can stop broker server by one of the following lines:
	
	```
	sudo service mosquitto stop
	sudo systemctl stop mosquitto.service
	```

	If you want to see verbose information, which is not displayed at CLI when broker is started as service,
	have a look on the corresponding log file:
	
	```
	sudo tail -n20 -f /var/log/mosquitto/mosquitto.log
	```

1. If desired, subscribe from second CLI for displaying all messages:

    ```
    mosquitto_sub -t "#" -v -u testuser
    ```
    
Here, relevant tags represent as follows:

- `h`: mqtt host to connect to. Defaults to localhost.
- `p`: network port to connect to. Defaults to 1883 for plain MQTT and 8883 for MQTT over TLS.
- `t`: mqtt topic to subscribe to. May be repeated multiple times.
- `u`: provide a username.
- `P`: provide a password.
- `v`: print published messages verbosely.

Alternatively, consider testing at world-wide, public test server:

    ```
	mosquitto_sub -t "CoNM/workflow_system" -v -u testuser -h "test.mosquitto.org" -p 1883
    ```

1. Start `AI_simulation_basis_communication_client.py` from a third CLI:

    ```
    python AI_simulation_basis_communication_client.py
    ```

1. Send message manually from fourth CLI:

    ```
    mosquitto_pub -t "CoNM/workflow_system" -u testuser -m "This is a test message."
    ```
    
Alternatively, consider testing at world-wide, public test server:

    ```
    mosquitto_pub -t "CoNM/workflow_system" -u testuser -m "Please do manaul test instruction mxy." -h "test.mosquitto.org" -p 1883
    ```
    
Face communication management throughout the process at first CLI being the `message broker`.
Relevant messages from subscriptions e.g. can be seen at the second CLI representing a machine being a `mqtt client`.
The third CLI represents a machine that is activated by messages in order to realize any kind of complex, algorithmic procedures.

### 5) Install relevant content manually

1. If further packages are required, you might install them from inside the container.

    ```
    python -m pip install -U pip
    python -m pip install -U matplotlib
    pip install SciPy
    ```
    
    Of course, you can prepare an corresponding image providing packages required.
    For a quick testing, this is okay.

## Use Cases

### A) Deploy relevant files from desktop to raspberry 

#### Deploy relevant files via terminal and scp command.
    
1. Copy relevant files from local desktop to raspberry.    
    
    ```
    scp -r /Users/mgrum/repositories/tensorflowWorkDir/ pi@141.89.39.173:/home/pi/knowledgeBases/
    ```
    
    Alternatively, you can deploy them via github or via docker.
    
#### Deploy relevant file, such as a solution/image, via docker and copy files manually.

1. Create Dockerfile for creating an image.

    ```
    # syntax=docker/dockerfile:1
    FROM busybox
    ADD ./apple_banana_orange_pump_20.h5  /knowledgeBase/currentSolution.h5
    CMD ["/knowledgeBase"]
    ```
    
    Alternatively, consider the following
    
    ```
    scratch       # 34.0MB
    busybox       # 35.2MB
    alpine:3.14   # 39.6MB
    ```

1. Build docker image from Dockerfile specified.

    ```
    docker build --tag apple_banana_orange_pump_20 .
    ```

1. Have a look on the image created.    
    
    ```
    docker run -it --rm apple_banana_orange_pump_20 sh
    ```
    
1. Copy data from container, e.g. this is called `bold_galileo`, to local file.

    ```
    docker cp bold_galileo:/test /Users/mgrum/repositories/tensorflowWorkDir/test_volume02
    ```

1. Start container having this directory mounted.

    ```
    docker run -it --rm \
    -v $PWD/repositories/tensorflowWorkDir:/tmp \
    -v /Users/mgrum/repositories/tensorflowWorkDir/test_volume02:/tmp/test_volume02 \
    -w /tmp \
    tensorflow/tensorflow
    ```
    
#### Deploy relevant file, such as a solution/image, via docker and consider file copying via `docker-compose`routines.

1. Follow individual readme files of `image` and `scenario` folders.


### B) Test installation of individual components

1. Test AI container manually:

	```
	cd $repositories/AI-CPS/scenarios/apply_knnSolution/x86_gpu
	docker-compose up
	```

1. Run local MQTT client:

	```
	cd $repositories/AI-CPS/code/
	sudo python3 AI_simulation_basis_communication_client.py
	```

1. Release one of the remote requests shown before and face AI case realization.

### C) Submit ANN requests from remote

1. Initiate example `apply_knnSolution` from remote CLI:

    ```
    mosquitto_pub -t "CoNM/workflow_system" -u testuser -m "Please realize the following AI case: scenario=apply_knnSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_20, activation_base=marcusgrum/activationbase_apple_okay_01, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=-, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    ```

1. Initiate example `create_knnSolution` from remote CLI:
    
    ```
    mosquitto_pub -t "CoNM/workflow_system" -u testuser -m "Please realize the following AI case: scenario=create_knnSolution, knowledge_base=-, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    ```

1. Initiate example `refine_knnSolution` from remote CLI:   
 
    ```
    mosquitto_pub -t "CoNM/workflow_system" -u testuser -m "Please realize the following AI case: scenario=refine_knnSolution, knowledge_base=marcusgrum/knowledgebase_apple_banana_orange_pump_01, activation_base=-, code_base=marcusgrum/codebase_ai_core_for_image_classification, learning_base=marcusgrum/learningbase_apple_banana_orange_pump_02, sender=SenderA, receiver=ReceiverB." -h "test.mosquitto.org" -p 1883
    ```

## Application examples

### Realize virtual experiments

Virtual / computer-based simulated experiment examples can be found at this repository at `./code/experiments`.

For instance, ...

- `experiment01` simulates the manipulation of CPS knowledge base by process change - alternative product change (in context of continual learning and training data manipulation): Inhowfar do AI-based CPS forget if a process changes leads to a different product type distribution (in this case alternative products) that does not correspond to the CPS's current specialization?

- `experiment02` simulates the manipulation of CPS knowledge base by smart sensors overtaking tasks partly - filtering sensory input (in context of continual learning and training data manipulation): Inhowfar do AI-based CPS forget if fruit evaluation is overtakten by preceeding smart sensors and the CPS's current specialization does not correspond to the required set of skills?

- `experiment03` simulates the manipulation of CPS knowledge base by worsening sensors badly affecting tasks: filtering sensory input (in context of continual learning and training data manipulation): Inhowfar do AI-based CPS forget if current evaluation task is disturbed by defect sensors, such as providing blurred images that do not correspond to the CPS's current specialization?

- `experiment04` simulates the manipulation of CPS knowledge base by process change - sub-set product change (in context of continual learning and training data manipulation): Inhowfar do AI-based CPS forget if a process changes leads to a different product type distribution (in this case a sub-set of trained products) that does not correspond to the CPS's current specialization? 

Here, AI-based CPS can carry out tasks in simulation runs and their performance can be measured.
As these are considered as AI-based individuals, statistical analyses can explore their performance.

### EV3-Based Industry4.0-Production-System

For instance, you can base an entire cyber-physical production system, 
such as the CoNM application example of `EV3-Based Industry4.0-Production-System`,
on the basis of AI-CPS.

Further details can be found at:

Grum M. 2022. Construction of a Concept of Neuronal Modeling. Springer Gabler Wiesbaden. https://doi.org/10.1007/978-3-658-35999-7

### Research and Application Center Industry 4.0

For instance, you can include AI-CPS in traditional cyber-physical production systems, 
such as the Research and Application Center Industry 4.0 (RACI).

<img src="./documentation/UMLsequenceDiagram.png" height="600" />

Here, a `NodeRed environment` can manage the AI knowledge base selection,
such as by an `AI Case-Based Reasoning`. 
The `AI simulation basis environment` carries out the ANN-based model application
and communicates results back to interested parties via the MQTT messaging broker.

Further details can be found at:

Grum, M. et al. 2023. AI Case-Based Reasoning for Artificial Neural Networks. A2IA 2023 Conference.

Applying this in ordinary production processes, experiments can be realized that focus on human-AI teaming.
The following example focuses on (1) human trust, (2) human frustation as well as (3) human forgetting because of AI support.

<img src="./documentation/Human-AI-Experiment01.png" height="600" />

As this Neuronal Modeling and Description Language (NMDL) model shows, 
humans are trained remotely to realize a production process.
Then, humans are faced with the trained production process within the real production environment.
Thereafter, humans are faced with AI support at the trained production process within the real production environment.
Lastly, humans are faced with the original production process (without AI support).
In between, production performance as well as trust and frustration are surveyed.

Experiments and publications about this are in progress.