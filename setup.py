import cv2
import time
import json
import os
import numpy as np
import paho.mqtt.client as mqtt 
from gpiozero import LEDBoard
from ultralytics import YOLO

MQTT_BROKER = os.environ['MQTT_BROKER']
MQTT_PORT = 1883
MQTT_USERNAME = os.environ['MQTT_USERNAME']
MQTT_PASSWORD = os.environ['MQTT_PASSWORD']
DEVICE_IP_ADDRESS = os.environ['DEVICE_IP_ADDRESS']
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"

# inisialisasi kamera model gpio pin
model_pose = YOLO("yolo11n-pose_ncnn_model", task="pose") 

cam_source = "usb0"
resW, resH = 640, 480
leds = LEDBoard(19, 26)

consecutive_detections = 0
gpio_state = 0 # 0 = OFF, 1 = ON. 
is_person_reported = False 
fps_buffer = []
fps_avg_len = 50

# Inisialisasi MQTT Client 
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()