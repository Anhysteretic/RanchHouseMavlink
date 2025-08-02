import cv2 as cv
import numpy as np
import threading
import math
import transformations
from CoordTransforms import CoordTransforms
from mavsdk import offboard
from detector import Detector
from simple_pid import PID
import time

IMAGE_HEIGHT, IMAGE_WIDTH = 720, 1280
y_cut = IMAGE_HEIGHT/2
_MAX_CLIMB, _MAX_SPEED, _MAX_YAWSPEED = 0.5, 0.5, 10
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2])
fixed_distance = 100

class Tracker():
    def __init__(self, _detector: Detector):
        
        self.latestControl: offboard.VelocityBodyYawspeed = offboard.VelocityBodyYawspeed(0, 0, 0, 0)
        
        self._detector = _detector
        
        self.running = False
        self.trackerThread = None
        self.threading_lock = threading.Lock()

        self.pid_x = PID(Kp=0.005, Ki=0.000, Kd=0, setpoint=0,
                         sample_time=0.1,
                         output_limits=(-_MAX_SPEED, _MAX_SPEED))

        self.pid_y = PID(Kp=0.005, Ki=0.000, Kd=0, setpoint=0,
                         sample_time=0.1,
                         output_limits=(-_MAX_SPEED, _MAX_SPEED))
        
        self.pid_yaw = PID(Kp=0.01, Ki=0.00, Kd=0.0, setpoint=0, 
                            sample_time=0.1, output_limits=(-_MAX_YAWSPEED, _MAX_YAWSPEED))
        
        return
    
    def startTrackerThread(self):
        self.running = True
        self.trackerThread = threading.Thread(target=self._proccessLoop, daemon=True)
        self.trackerThread.start()
        
    def stopTrackerThread(self):
        if self.running:
            self.running = False
            self.trackerThread.join()

    def getLatestControl(self) -> offboard.VelocityBodyYawspeed:
        with self.threading_lock: return self.latestControl
        
    def _proccessLoop(self):
        
        while self.running:            
            local_detector_result = self._detector.getLatestResult()

            if local_detector_result is None: continue 

            try:
                line1_p1 = np.array([local_detector_result[0][0], local_detector_result[0][1]])
                line1_p2 = np.array([local_detector_result[0][2], local_detector_result[0][3]])
                
                line2_p1 = np.array([local_detector_result[1][0], local_detector_result[1][1]])
                line2_p2 = np.array([local_detector_result[1][2], local_detector_result[1][3]])
            except Exception:
                continue
            
            seg1_p1, seg1_p2 = self.truncateLine(line1_p1, line1_p2)
            seg2_p1, seg2_p2 = self.truncateLine(line2_p1, line2_p2)

            len1 = np.linalg.norm(seg1_p2 - seg1_p1) if seg1_p1 is not None else 0
            len2 = np.linalg.norm(seg2_p2 - seg2_p1) if seg2_p1 is not None else 0

            if len1 == 0 and len2 == 0: continue

            if len1 >= len2: p_start, p_end = seg1_p1, seg1_p2
            else: p_start, p_end = seg2_p1, seg2_p2

            V = p_end - p_start
            S = p_start - CENTER

            a = np.dot(V, V)
            if a == 0: continue
                
            b = 2 * np.dot(S, V)
            c = np.dot(S, S) - fixed_distance**2

            delta = b**2 - 4 * a * c
            if delta < 0: continue
            
            valid_t = None
            t1 = (-b - np.sqrt(delta)) / (2 * a)
            t2 = (-b + np.sqrt(delta)) / (2 * a)
                        
            if 0 <= t1 <= 1: valid_t = t1
            elif 0 <= t2 <= 1: valid_t = t2
            
            if valid_t is None: continue

            Q = p_start + valid_t * V
            final_vector = Q - CENTER

            error_x = final_vector[0]
            error_y = final_vector[1]

            angle_radians = math.atan2(final_vector[1], final_vector[0])
            angle_degrees = math.degrees(angle_radians)

            vx_body = self.pid_y(error_y)
            vy_body = -self.pid_x(error_x)
            vz_body = 0.0
            yaw_speed = self.pid_yaw(angle_degrees)

            with self.threading_lock: self.latestControl = offboard.VelocityBodyYawspeed(
                        vx_body,
                        vy_body,
                        vz_body,
                        yaw_speed
                    )

    def truncateLine(self, p1, p2):
        y1, y2 = p1[1], p2[1]      
        if y1 >= y_cut and y2 >= y_cut: return None, None      
        if y1 < y_cut and y2 < y_cut: return p1, p2
        if y1 >= y_cut:
            p1, p2 = p2, p1
            y1, y2 = p1[1], p2[1]    
        t = (y_cut - y1) / (y2 - y1) # parametric equation
        p_int = p1 + t * (p2 - p1)
        return p1, p_int
                
