from hardware.servo_controller import ServoController
from hardware.config import SERVO_CHANNELS

print("SpotMicro 12-servo configuration")
print("--------------------------------")

for leg, joints in SERVO_CHANNELS.items():
    print(f"{leg}:")
    for joint, channel in joints.items():
        print(f"  {joint}: PCA9685 channel {channel}")

print("--------------------------------")
print("12 servo channels configured.")
print("PCA9685 address: 0x40")
print("Servo frequency: 50 Hz")
