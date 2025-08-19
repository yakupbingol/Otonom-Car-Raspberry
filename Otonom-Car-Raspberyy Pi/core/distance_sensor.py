
import RPi.GPIO as GPIO
import time


class DistanceSensor:

    def __init__(self) -> None:
        self.gpio = GPIO
        self.trig_pin = 16
        self.echo_pin = 18
        self.distance = None
        self.setup()

    def setup(self):

        self.gpio.setmode(self.gpio.BOARD)
        self.gpio.setup(self.trig_pin, self.gpio.OUT)
        self.gpio.setup(self.echo_pin, self.gpio.IN)

    def track(self):

        self.gpio.output(self.trig_pin, self.gpio.LOW)
        time.sleep(0.2)
        self.gpio.output(self.trig_pin, self.gpio.HIGH)
        time.sleep(0.00001)
        self.gpio.output(self.trig_pin, self.gpio.LOW)

        while self.gpio.input(self.echo_pin) == 0:
            pulse_start = time.time()

        while self.gpio.input(self.echo_pin) == 1:
            pulse_end = time.time()

        pulse_duration = pulse_end - pulse_start
        distance = pulse_duration * 17150
        distance = round(distance, 2)
        self.distance = distance
        time.sleep(0)

    def stop(self):
        self.gpio.cleanup()

    def get_distance(self):
        return self.distance
