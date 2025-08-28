import cv2
import time
import json
import numpy as np
import paho.mqtt.client as mqtt
from gpiozero import LED
from ultralytics import YOLO
from datetime import datetime

MQTT_BROKER = "192.168.0.174"
MQTT_PORT = 1883
MQTT_USERNAME = "cpsmagang"
MQTT_PASSWORD = "cpsjaya123"
DEVICE_IP_ADDRESS = "192.168.0.174"

LAMP_PIN = 26
FAN_PIN = 19
CAM_SOURCE = 0  #
CAM_WIDTH = 640
CAM_HEIGHT = 480
YOLO_MODEL_PATH = "yolov8n-pose.pt"

PROCESS_EVERY_N_FRAMES = 5
DETECTION_THRESHOLD_ON = 8
DETECTION_THRESHOLD_OFF = 2

STATUS_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/status"
SENSOR_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/sensor"
ACTION_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/action"
SETTINGS_UPDATE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/settings/update"
MODE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/mode"
SCHEDULE_TOPIC = f"iot/{DEVICE_IP_ADDRESS}/schedule"

model_pose = YOLO(YOLO_MODEL_PATH)
lamp = LED(LAMP_PIN, active_high=False) 
fan = LED(FAN_PIN, active_high=False)

cam = cv2.VideoCapture(CAM_SOURCE)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
if not cam.isOpened():
    print("Gagal membuka kamera.")
    exit()

device_states = {"lamp": 0, "fan": 0}  # 0=OFF, 1=ON
current_mode = "auto"  # default mode
schedule = {"start": "18:00", "end": "22:00"}
consecutive_detections = 0
is_person_reported = False
frame_counter = 0
last_pose_found = False

# Function to control devices
def control_device(device, action):
    global device_states
    target = lamp if device == "lamp" else fan if device == "fan" else None
    if target:
        if action == "turn_on":
            target.on()
            device_states[device] = 1
        elif action == "turn_off":
            target.off()
            device_states[device] = 0
        publish_status()
        print(f"Manual control: {device} -> {action}")

def is_within_schedule():
    now = datetime.now().strftime("%H:%M")
    return schedule["start"] <= now <= schedule["end"]

def publish_status():
    payload = json.dumps({
        "mode": current_mode,
        "lamp": "ON" if device_states["lamp"] else "OFF",
        "fan": "ON" if device_states["fan"] else "OFF"
    })
    client.publish(STATUS_TOPIC, payload, qos=1, retain=True)

# callback
def on_connect(client, userdata, flags, rc, properties=None):
    if rc == 0:
        print(f"Connected to MQTT Broker: {MQTT_BROKER}")
        client.subscribe(ACTION_TOPIC)
        client.subscribe(SETTINGS_UPDATE_TOPIC)
        client.subscribe(MODE_TOPIC)
        client.subscribe(SCHEDULE_TOPIC)
        publish_status()
    else:
        print(f"Failed to connect, code: {rc}")

def on_message(client, userdata, msg):
    global current_mode, schedule
    try:
        payload = json.loads(msg.payload.decode())
    except:
        print("Invalid payload")
        return

    if msg.topic == ACTION_TOPIC and current_mode == "manual":
        device = payload.get("device")
        action = payload.get("action")
        if device and action:
            control_device(device, action)

    elif msg.topic == MODE_TOPIC:
        new_mode = payload.get("mode")
        if new_mode in ["manual", "auto", "schedule"]:
            current_mode = new_mode
            print(f"Mode changed to {current_mode}")
            publish_status()

    elif msg.topic == SCHEDULE_TOPIC:
        if "start" in payload and "end" in payload:
            schedule["start"] = payload["start"]
            schedule["end"] = payload["end"]
            print(f"Schedule updated: {schedule}")

# setup mqtt nya
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
client.on_connect = on_connect
client.on_message = on_message
last_will_payload = json.dumps({"status": "offline"})
client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)
client.connect(MQTT_BROKER, MQTT_PORT, 60)
client.loop_start()

# auto sched kontrol
print("\nSistem berjalan. Tekan 'Q' untuk berhenti.")
try:
    while True:
        ret, frame = cam.read()
        if not ret:
            time.sleep(1)
            continue

        annotated_frame = frame.copy()

        if current_mode == "auto":
            frame_counter += 1
            if frame_counter % PROCESS_EVERY_N_FRAMES == 0:
                results = model_pose.predict(frame, verbose=False)
                last_pose_found = len(results) > 0 and results[0].keypoints is not None

            if last_pose_found:
                consecutive_detections = min(consecutive_detections + 1, 10)
            else:
                consecutive_detections = max(consecutive_detections - 1, 0)

            should_be_active = consecutive_detections >= DETECTION_THRESHOLD_ON
            should_be_inactive = consecutive_detections <= DETECTION_THRESHOLD_OFF

            if should_be_active:
                if device_states["lamp"] == 0: lamp.on(); device_states["lamp"] = 1
                if device_states["fan"] == 0: fan.on(); device_states["fan"] = 1
            elif should_be_inactive:
                if device_states["lamp"] == 1: lamp.off(); device_states["lamp"] = 0
                if device_states["fan"] == 1: fan.off(); device_states["fan"] = 0

        elif current_mode == "schedule":
            if is_within_schedule():
                if device_states["lamp"] == 0: lamp.on(); device_states["lamp"] = 1
                if device_states["fan"] == 0: fan.on(); device_states["fan"] = 1
            else:
                if device_states["lamp"] == 1: lamp.off(); device_states["lamp"] = 0
                if device_states["fan"] == 1: fan.off(); device_states["fan"] = 0

        # Display info
        mode_text = f"Mode: {current_mode.upper()}"
        cv2.putText(annotated_frame, mode_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 0), 2)

        cv2.imshow("Smart Home", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    client.publish(STATUS_TOPIC, json.dumps({"status": "offline"}), retain=True)
    cam.release()
    cv2.destroyAllWindows()
    lamp.close()
    fan.close()
    client.loop_stop()
    client.disconnect()
    print("Selesai.")
