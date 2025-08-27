from gpiozero import LED

class DeviceController:
    def __init__(self):
        self.lamp = LED(26)
        self.fan = LED(19)
        self.device_auto_modes = {"lamp": True, "fan": True}
        self.lamp_state = 0
        self.fan_state = 0

    def control_device(self, device, action):
        if device == "lamp":
            if action == "turn_on": self.lamp.on(); self.lamp_state = 1
            else: self.lamp.off(); self.lamp_state = 0
        elif device == "fan":
            if action == "turn_on": self.fan.on(); self.fan_state = 1
            else: self.fan.off(); self.fan_state = 0
        if device in self.device_auto_modes:
            self.device_auto_modes[device] = False
            print(f"'{device}' Mode Manual.")

    def auto_control(self, should_be_active, should_be_inactive):
        if should_be_active:
            if self.device_auto_modes["lamp"] and self.lamp_state == 0:
                self.lamp_state = 1; self.lamp.on()
            if self.device_auto_modes["fan"] and self.fan_state == 0:
                self.fan_state = 1; self.fan.on()
        elif should_be_inactive:
            if self.device_auto_modes["lamp"] and self.lamp_state == 1:
                self.lamp_state = 0; self.lamp.off()
            if self.device_auto_modes["fan"] and self.fan_state == 1:
                self.fan_state = 0; self.fan.off()

    def cleanup(self):
        self.lamp.close()
        self.fan.close()
