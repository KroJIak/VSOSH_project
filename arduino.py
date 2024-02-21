import serial
from time import sleep

class Arduino:
    def __init__(self, port, baudrate=9600, log=True, timeout=None, coding='utf-8'):
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=timeout, write_timeout=1)
        self.coding = coding
        self.log = log
        sleep(2)
        self.serial.flush()

    def sendData(self, data: str):
        if self.log: print('Sent to Arduino:', data)
        data += '\n'
        self.serial.write(data.encode(self.coding))

    def setRedLight(self):
        msg = f'LIGHT.RED'
        self.sendData(msg)

    def setGreenLight(self):
        msg = f'LIGHT.GREEN'
        self.sendData(msg)

    def setAngle(self, angle):
        msg = f'SERVO.SET:{angle}'
        self.sendData(msg)
