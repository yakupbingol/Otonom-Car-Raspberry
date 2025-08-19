
from .dc_motor import DcMotor
from .servo import Servo_Motor


class Car:

    def __init__(self, dc_motor: DcMotor, servo_motor: Servo_Motor) -> None:
        self.dc_motor: DcMotor = dc_motor
        self.servo_motor: Servo_Motor = servo_motor

    def forward(self):
        self.dc_motor.power_on()

    def stop(self):
        self.dc_motor.power_off()

    def back(self):
        self.dc_motor.power_reverse_on()

    def rotate(self, angle):
        self.servo_motor.rotate_servo(angle)
