import time
import RPi.GPIO as GPIO


class Servo_Motor:

    def __init__(self, servo_pin=29) -> None:
        self.gpio = GPIO
        self.gpio.setmode(self.gpio.BOARD)
        self.gpio.setup(servo_pin, self.gpio.OUT)
        self.pwm = self.gpio.PWM(servo_pin, 50)

    def start(self):
        self.pwm.start(0)

    def calculate_duty_cycle(self, angle):
        duty_cycle = 2.5 + (angle/18)
        return duty_cycle

    def clean_angle(self):
        self.pwm.ChangeDutyCycle(7.5)

    def stop(self):
        self.pwm.stop()
        self.gpio.cleanup()

    def rotate_servo(self, angle, in_area):

        step = 5
        delay = 0.1
        last = 90
        direction = 1 if angle > last else -1

        for ang in range(int(last), int(angle), step * direction):

            current_duty_cycle = 2.5 + (ang/18)
            final_duty = 0

            if in_area:
                self.clean_angle()
                break

            if current_duty_cycle < 5.83333:
                final_duty = 5.83333
            if current_duty_cycle > 9.16666667:
                final_duty = 9.16666667
            if 5.83333 <= current_duty_cycle <= 9.16666667:
                final_duty = current_duty_cycle

            self.pwm.ChangeDutyCycle(final_duty)
            time.sleep(delay)
        last = angle











