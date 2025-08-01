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
_MAX_CLIMB, _MAX_SPEED, _MAX_YAWSPEED = 0.5, 0.5, 0.5
CENTER = np.array([IMAGE_WIDTH//2, IMAGE_HEIGHT//2])
fixed_distance = 100

class Tracker():
    def __init__(self, _detector: Detector):
        
        self.latestControl: offboard.VelocityBodyYawspeed = offboard.VelocityBodyYawspeed(0, 0, 0, 0)
        
        self._detector = _detector
        
        self.running = False
        self.trackerThread = None
        self.threading_lock = threading.Lock()

        self.pid_x = PID(Kp=0.01, Ki=0.000, Kd=0.005, setpoint=0,
                         sample_time=0.1,
                         output_limits=(-_MAX_SPEED, _MAX_SPEED))

        self.pid_y = PID(Kp=0.01, Ki=0.000, Kd=0.005, setpoint=0,
                         sample_time=0.1,
                         output_limits=(-_MAX_SPEED, _MAX_SPEED))
        
        self.pid_yaw = PID(Kp=0.01, Ki=0.001, Kd=0.005, setpoint=0, 
                            sample_time=0.1, output_limits=(-_MAX_YAWSPEED, _MAX_YAWSPEED))
        
        return
    
    def startTrackerThread(self):
        self.running = True
        print("[DEBUG] Starting tracker thread...")
        self.trackerThread = threading.Thread(target=self._proccessLoop, daemon=True)
        self.trackerThread.start()
        print("[DEBUG] Tracker thread started successfully")
        
    def stopTrackerThread(self):
        if self.running:
            print("[DEBUG] Stopping tracker thread...")
            self.running = False
            self.trackerThread.join()
            print("[DEBUG] Tracker thread stopped successfully")
        else:
            print("[DEBUG] Tracker thread was not running")

    def getLatestControl(self) -> offboard.VelocityBodyYawspeed:
        with self.threading_lock: return self.latestControl
        
    def _proccessLoop(self):
        loop_count = 0
        tracker_start_time = time.time()
        print("[DEBUG] Tracker processing loop started")
        
        while self.running:
            loop_start_time = time.time()
            loop_count += 1
            
            local_detector_result = self._detector.getLatestResult()
            
          
            if local_detector_result is None:
                if loop_count % 100 == 0:  # Print every 100 loops to avoid spam
                    print("[DEBUG] No detector result available")
                continue

            print(f"[DEBUG] Processing lines: {local_detector_result}")

            line1_p1 = np.array([local_detector_result[0][0], local_detector_result[0][1]])
            line1_p2 = np.array([local_detector_result[0][2], local_detector_result[0][3]])
            
            line2_p1 = np.array([local_detector_result[1][0], local_detector_result[1][1]])
            line2_p2 = np.array([local_detector_result[1][2], local_detector_result[1][3]])
            
            seg1_p1, seg1_p2 = self.truncateLine(line1_p1, line1_p2)
            seg2_p1, seg2_p2 = self.truncateLine(line2_p1, line2_p2)

            print(f"[DEBUG] Truncated segments - Line1: {seg1_p1} to {seg1_p2}, Line2: {seg2_p1} to {seg2_p2}")

            len1 = np.linalg.norm(seg1_p2 - seg1_p1) if seg1_p1 is not None else 0
            len2 = np.linalg.norm(seg2_p2 - seg2_p1) if seg2_p1 is not None else 0

            print(f"[DEBUG] Line lengths - Line1: {len1:.1f}, Line2: {len2:.1f}")

            if len1 == 0 and len2 == 0: 
                print("[DEBUG] No valid line segments after truncation")
                continue

            if len1 >= len2: 
                p_start, p_end = seg1_p1, seg1_p2
                print(f"[DEBUG] Using Line1 (longer): {p_start} to {p_end}")
            else: 
                p_start, p_end = seg2_p1, seg2_p2
                print(f"[DEBUG] Using Line2 (longer): {p_start} to {p_end}")

            V = p_end - p_start
            S = p_start - CENTER

            a = np.dot(V, V)
            if a == 0: 
                print("[DEBUG] Line vector has zero length")
                continue
                
            b = 2 * np.dot(S, V)
            c = np.dot(S, S) - fixed_distance**2

            delta = b**2 - 4 * a * c
            if delta < 0:
                print(f"[DEBUG] No intersection with circle (delta={delta:.2f})")
                continue
            
            valid_t = None
            t1 = (-b - np.sqrt(delta)) / (2 * a)
            t2 = (-b + np.sqrt(delta)) / (2 * a)
            
            print(f"[DEBUG] Intersection parameters: t1={t1:.3f}, t2={t2:.3f}")
            
            if 0 <= t1 <= 1:
                valid_t = t1
                print(f"[DEBUG] Using t1={t1:.3f}")
            elif 0 <= t2 <= 1:
                valid_t = t2
                print(f"[DEBUG] Using t2={t2:.3f}")
            
            if valid_t is None: 
                print("[DEBUG] No valid intersection point on line segment")
                continue

            Q = p_start + valid_t * V
            final_vector = Q - CENTER

            error_x = final_vector[0]
            error_y = final_vector[1]

            angle_radians = math.atan2(final_vector[1], final_vector[0])
            angle_degrees = math.degrees(angle_radians)

            print(f"[DEBUG] Target point Q: {Q}, Error: ({error_x:.1f}, {error_y:.1f}), Angle: {angle_degrees:.1f}°")

            vx_body = self.pid_y(error_y)
            vy_body = -self.pid_x(error_x)
            vz_body = 0.0
            yaw_speed = self.pid_yaw(angle_degrees)

            print(f"[DEBUG] Control outputs - vx: {vx_body:.3f}, vy: {vy_body:.3f}, yaw_speed: {yaw_speed:.3f}")

            with self.threading_lock: self.latestControl = offboard.VelocityBodyYawspeed(
                        vx_body,
                        vy_body,
                        vz_body,
                        yaw_speed
                    )
                
            loop_processing_time = time.time() - loop_start_time
            
            # Print summary every 10 successful processing loops
            if loop_count % 10 == 0:
                total_runtime = time.time() - tracker_start_time
                avg_loop_time = total_runtime / loop_count
                print(f"[DEBUG] Loop {loop_count}: Processing time: {loop_processing_time*1000:.1f}ms, "
                        f"Avg loop time: {avg_loop_time*1000:.1f}ms, Total runtime: {total_runtime:.1f}s")
            
            elapsed_time = time.time() - loop_start_time
            sleep_time = self.pid_x.sample_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
                    
        total_runtime = time.time() - tracker_start_time
        print(f"[DEBUG] Tracker loop finished. Total runtime: {total_runtime:.2f}s, "
              f"Processed {loop_count} loops, Avg loop time: {total_runtime/loop_count*1000:.1f}ms")

    def truncateLine(self, p1, p2):
        y1, y2 = p1[1], p2[1]
        
        if y1 >= y_cut and y2 >= y_cut: 
            print(f"[DEBUG] Line fully below y_cut ({y_cut}): ({y1:.1f}, {y2:.1f}) -> None")
            return None, None
            
        if y1 < y_cut and y2 < y_cut: 
            print(f"[DEBUG] Line fully above y_cut ({y_cut}): ({y1:.1f}, {y2:.1f}) -> keeping full line")
            return p1, p2
            
        if y1 >= y_cut:
            p1, p2 = p2, p1
            y1, y2 = p1[1], p2[1]
            print(f"[DEBUG] Swapped points for truncation")
            
        t = (y_cut - y1) / (y2 - y1) # parametric equation
        p_int = p1 + t * (p2 - p1)
        print(f"[DEBUG] Line truncated at y_cut: t={t:.3f}, intersection={p_int}")
        return p1, p_int
                