# AI-CPS



## Getting Started

### Set up device via terminal

1. Download and install recent OS, e.g. `raspberry Pi OS Lite buster 12.02.2020`, on sd card via laptop.

1. Create ssh file an boot section.

1. Configure WLAN information.

1. Put SD card into raspberry and start device.

### Alternatively, set up device via Raspberry Pi Imager

1. Download and install Raspberry Pi Imager (e.g. v1.7.2) from [Raspberry Pi OS Page](https://www.raspberrypi.com/software/).

1. Configure your device by selecting `rasperry Pi OS Lite (64bit)`, SD card and corresponding settings.

1. Hit `write os`.

1. Put SD card into raspberry and start device.

### Log in to your device

1. Change default password.

    ```
    sudo passwd pi
    ```

1. Connect on your raspberry via shell with user `pi`, e.g. with password `RaspBerry`:

    ```
    ssh pi@141.89.39.173
    ```

    or by

    ```
    ssh pi@AiLabraspberry1.local
    ```

1. Test your current distro by `lsb_release -a`.

### Set up docker

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

### Set up tensorflow
    
    https://github.com/fgervais/docker-tensorflow/blob/master/Dockerfile
    https://github.com/armindocachada/raspberrypi-docker-tensorflow-opencv
    https://github.com/lhelontra/tensorflow-on-arm
    https://github.com/samjabrahams/tensorflow-on-raspberry-pi
    
    Camera
    https://www.raspberrypi.com/products/raspberry-pi-high-quality-camera/

#### Build own tensorflow containters

    https://github.com/tensorflow/build
    https://medium.com/tensorflow/tensorflow-1-9-officially-supports-the-raspberry-pi-b91669b0aa0
    https://www.tensorflow.org/install/docker
    https://github.com/tensorflow/build/tree/master/raspberry_pi_builds

#### Use tensorflow container to run in an interactive bash
    
1. Start container.

    ```
    docker run --rm -it francoisgervais/tensorflow:2.1.0-cp35 bash
    ```   
    
    Alternatively, use these containers
    
    ```
    armswdev/tensorflow-arm-neoverse
    ```
    
#### Use tensorflow container to run a small program

1. Run tensorflow docker container and execute a small example script.

    ```
    docker run -it francoisgervais/tensorflow:2.1.0-cp35 python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
    ```

#### Use tensorflow container to run a script

1. Run tensorflow docker container and execute a small example script.

    ```
    docker run -it francoisgervais/tensorflow:2.1.0-cp35 python3 -c script_file
    ```