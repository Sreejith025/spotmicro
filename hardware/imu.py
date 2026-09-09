import board
import busio
import adafruit_bno055


class IMU:
    def __init__(self):
        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_bno055.BNO055_I2C(i2c)

    def read(self):
        return {
            "temperature": self.sensor.temperature,
            "acceleration": self.sensor.acceleration,
            "gyro": self.sensor.gyro,
            "magnetic": self.sensor.magnetic,
            "euler": self.sensor.euler,
            "quaternion": self.sensor.quaternion,
        }


if __name__ == "__main__":
    print("BNO055 IMU module ready.")
