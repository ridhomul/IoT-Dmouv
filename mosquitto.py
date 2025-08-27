import json
import paho.mqtt.client as mqtt
from setup import *

class MQTTClient:
    def __init__(self, device_controller):
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
        self.device_controller = device_controller

        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        last_will_payload = json.dumps({"status": "offline"})
        self.client.will_set(STATUS_TOPIC, payload=last_will_payload, qos=1, retain=True)

    def connect(self):
        self.client.connect(MQTT_BROKER, MQTT_PORT, 60)
        self.client.loop_start()

    def on_connect(self, client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"Terhubung ke MQTT Broker {MQTT_BROKER}!")
            client.subscribe(ACTION_TOPIC)
            client.subscribe(SETTINGS_UPDATE_TOPIC)
            status_payload = json.dumps({"status": "online"})
            client.publish(STATUS_TOPIC, status_payload)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if msg.topic == ACTION_TOPIC:
                device = payload.get("device")
                action = payload.get("action")
                if device and action:
                    self.device_controller.control_device(device, action)

            elif msg.topic == SETTINGS_UPDATE_TOPIC:
                device = payload.get("device")
                if device in self.device_controller.device_auto_modes and "auto_mode_enabled" in payload:
                    new_mode = payload["auto_mode_enabled"]
                    self.device_controller.device_auto_modes[device] = new_mode
                    mode_status = "DIAKTIFKAN" if new_mode else "DINONAKTIFKAN"
                    print(f"SETTINGS UPDATE: Mode Otomatis '{device}' {mode_status}")
        except Exception as e:
            print(f"Error {e}")

    def publish_sensor(self, data):
        self.client.publish(SENSOR_TOPIC, json.dumps(data))

    def disconnect(self):
        self.client.loop_stop()
        self.client.disconnect()
