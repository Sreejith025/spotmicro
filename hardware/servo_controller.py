import board
import busio
from adafruit_pca9685 import PCA9685


class ServoController:
    def __init__(self, address=0x40):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.pca = PCA9685(i2c, address=address)

        # Standard hobby-servo PWM frequency
        self.pca.frequency = 50

    def set_angle(self, channel, angle):
        angle = max(0, min(180, angle))

        # Approximate 500–2500 µs pulse range at 50 Hz
        min_pulse = 500
        max_pulse = 2500

        pulse_us = min_pulse + (angle / 180) * (max_pulse - min_pulse)

        period_us = 1_000_000 / self.pca.frequency
        duty_cycle = int((pulse_us / period_us) * 65535)

        self.pca.channels[channel].duty_cycle = duty_cycle

    def release(self, channel):
        self.pca.channels[channel].duty_cycle = 0

    def close(self):
        self.pca.deinit()


if __name__ == "__main__":
    print("PCA9685 servo controller module ready.")
