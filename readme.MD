# Dmouv

Dmouv is a home automation program using a Raspberry Pi connected to lights and fan
This code was written for a Raspberry Pi.

# Software Structure

The program parses a configuration file at start-up to set initial settings.
Separate classes are used for each category of devices, lights,
fan, status, and automatic/manual interaction (action). These classes provide methods for controlling and reading the state of these devices.
A timer is used to schedule on and off times for the lights and other devices. Schedule can be controlled by the main app

# Installation

This project was developed on a Raspberry Pi running
[Raspberry Pi OS](https://www.raspberrypi.org/software/operating-systems/)
(32-bit or 64-bit) and written in Python version 3.
The code relies heavily on [Mosquitto MQTT](https://mosquitto.org/)
to bridge a network of devices through MQTT (a common IoT networking protocol) and backend (express)

## Install Mosquitto

The first step is to install `mosquitto` which provides an open source MQTT broker.
This can be installed from the command-line as follows:

```bash
sudo apt install -y mosquitto mosquitto-clients
```

Since we will be connecting to the MQTT broker locally, we can edit the mosquitto
configuration file to explicitly listen _only_ on the local loopback interface.
This can be done by adding the following lines in `/etc/mosquitto/conf.d/local.conf`:

```bash
listener 1883 127.0.0.1
allow_anonymous true
```

Next, enable the `mosquitto` broker service as follows:

```bash
sudo systemctl enable mosquitto.service
```

Ensure the `mosquitto` service is now running by typing:

```bash
sudo service mosquitto status
```

## Setting up the Python control software

Once the MQTT Broker is installed and devices are successfully paired we can setup the
`all.py` control program itself. This program communicates with devices by
sending messages to the MQTT broker which are then bridged to our backend services.
The control program is written in Python version 3 and uses the
[paho-mqtt](https://www.eclipse.org/paho/index.php?page=clients/python/index.php) library to send
MQTT messages. The dependencies for `all.py` can all be installed from the command-line as follows:

```bash
sudo apt-get install python3-pip
pip3 install paho-mqtt
```

## Configuration

Dmouv includes a `Setup.py` configuration file which should be adjusted to reflect
your local settings. In particular, you will need to specify the "friendly names" of any devices, and lights you are using along with the IP Address of the MQTT broker for
reaching the backend services (ideally the broker will be run on the local host).
Furthermore, we can set the right time frame through this file.

## Launching the program

The program can be launched from the command-line from the installation folder as follows:

```bash
python3 all.py
```

The `all.py` program may also be automatically launched at boot time as a systemd service.
This service must be configured to wait for the network to come online before starting.
This can be configured by creating a systemd service file in `/etc/systemd/system/Dmouv.service`
with the following settings:

```bash
[Unit]
Description=all
After=network-online.target
[Service]
ExecStart=/bin/sh -c "/usr/bin/python3 all.py"
WorkingDirectory=/home/cps/Dmouv
Restart=always
User=cps
[Install]
WantedBy=multi-user.target
```

Note that the `User` and `WorkingDirectory` will need to be set to reflect
your default username and the directory where the Dmouv source files are installed.
Finally, enable the Dmouv service as follows:

```bash
sudo systemctl enable pi-home.service
```

Reboot the computer and ensure that the service is started as expected. To check the status
of the service, type:

```bash
sudo systemctl status pi-home.service
```
test akun