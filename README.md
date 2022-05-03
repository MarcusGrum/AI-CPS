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

This is based on the [Tutorial](https://dev.to/elalemanyo/how-to-install-docker-and-docker-compose-on-raspberry-pi-1mo).

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

1. Create non-root user `testUser`.

```
sudo adduser ${USER}
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

1. Run Hello World Container for testing Docker installation.

```
docker run hello-world
```