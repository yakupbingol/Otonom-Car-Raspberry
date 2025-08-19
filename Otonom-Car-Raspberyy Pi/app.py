import numpy as np
import asyncio
import base64
import websockets
import RPi.GPIO as GPIO

from lane_detector import processFrame
from core import DcMotor, Servo_Motor , DistanceSensor
from threading import Thread
from time import sleep

from models import ThreadPool
import cv2

class Processor:
    def __init__(self):
        self.websocket_uri = "ws://192.168.2.195:8765"
        GPIO.setwarnings(False)
        self.dc_motor: DcMotor = DcMotor()
        self.servo_motor: Servo_Motor = Servo_Motor()
        self.thread_pool: ThreadPool = ThreadPool()
        self.is_available = False
        self.is_thread_on = True
        self.distance_sensor : DistanceSensor = DistanceSensor()
        self.distance = 1000
        self.distance_threshold = 50
        self.is_lane_changing = False
        self.lane_changing_angle = 20


    def websocket_controller(self):
        try:
            async def send_video(websocket):
                while True:
                    if self.is_available:
                        frame = cv2.resize(self.output_image, (500, 500))
                        _, img_encoded = cv2.imencode('.jpg', frame)
                        img_data = base64.b64encode(img_encoded)
                        await websocket.send(img_data)

            async def main():
                async with websockets.connect(self.websocket_uri) as websocket:
                    video_task = asyncio.create_task(send_video(websocket))
                    await asyncio.gather(video_task)

            asyncio.run(main())


        except Exception as e:
            self.dc_motor.stop()
            self.servo_motor.stop()
            self.distance_sensor.stop()
        finally:
            self.dc_motor.stop()
            self.servo_motor.stop()
            self.distance_sensor.stop()
            


    def frame_controller(self):
        try :

            cap = cv2.VideoCapture(0)
            while cap.isOpened and self.is_thread_on:
                ret, frame = cap.read()
                if not ret:
                    self.is_available = False
                    self.is_thread_on = False
                    self.dc_motor.stop()
                    self.servo_motor.stop()
                    self.distance_sensor.stop()
                    break

                self.result1, self.result2, self.current_lane, self.calculated_angle, self.angle_direction, self.in_area,self.output_image = processFrame(frame)
                self.angle = self.calculated_angle * self.angle_direction + 90
                self.is_available = True

                if cv2.waitKey(1) and 0xFF == ord("q"):
                    self.is_available = False
                    self.is_thread_on = False
                    self.dc_motor.stop()
                    self.servo_motor.stop()
                    self.distance_sensor.stop()
                    break


        except Exception as e:
            self.is_available = False
            self.is_thread_on = False
            self.dc_motor.stop()
            self.servo_motor.stop()
            self.distance_sensor.stop()
        finally:
            self.is_available = False
            self.is_thread_on = False
            self.dc_motor.stop()
            self.servo_motor.stop()
            self.distance_sensor.stop()
     





    def servo_motor_controller(self):

        try:

            while self.is_thread_on:

                if self.is_available and not self.is_lane_changing:
                    self.servo_motor.start()
                    if not self.in_area:
                        self.servo_motor.rotate_servo(self.angle, self.in_area)
                    if self.in_area:
                        self.servo_motor.clean_angle()

        except Exception as e:
            self.servo_motor.stop()
        finally:
            self.servo_motor.stop()

    def dc_motor_controller(self):
        try:
            while self.is_thread_on:
                if self.is_available:
                    self.dc_motor.power_on()

        except Exception as e :
            self.dc_motor.stop()

        finally:
            self.dc_motor.stop()

    def lane_changing_base_distance(self, direction):
        self.servo_motor.rotate_servo(
            self.lane_changing_angle * direction + 90, False)
        sleep(1)
        self.servo_motor.rotate_servo(90, False)
        sleep(1)

    def distance_controller(self):
        try:

            while self.is_thread_on:
                self.distance_sensor.track()
                self.distance = self.distance_sensor.get_distance()
                if self.distance < self.distance_threshold:
                    self.is_lane_changing = True
                    cross_lane_multiplier = 1 if self.current_lane == "left" else -1
                    self.lane_changing_base_distance(cross_lane_multiplier)
                    self.is_lane_changing = False


        except Exception as e:
            self.distance_sensor.stop()
        finally:
            self.distance_sensor.stop()



    def start(self):
        self.thread_pool.add_thread(Thread(target=self.frame_controller))
        self.thread_pool.add_thread(Thread(target=self.websocket_controller))
        self.thread_pool.add_thread(Thread(target=self.dc_motor_controller))
        self.thread_pool.add_thread(Thread(target=self.servo_motor_controller))
        self.thread_pool.add_thread(Thread(target=self.distance_controller))
        self.thread_pool.start_threading()

if __name__ == "__main__":
    processor = Processor()
    processor.start()
