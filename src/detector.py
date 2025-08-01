#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import threading
import time
import sys

camera_device_path = '/dev/video0'
frame_width = 1280  # Width of the frame
frame_height = 720  # Height of the frame
fps = 30            # Frames per second
codec = cv.VideoWriter_fourcc(*'MJPG') # Codec for MJPEG video

class Detector():
    def __init__(self):        
        self.latest_image = None
        self.latest_results = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        self.thread_lock = threading.Lock()
        
        self.running = False
        self.videoThread = None

        self.cap = cv.VideoCapture(camera_device_path)
        if not self.cap.isOpened():
            print(f"Error: Could not open video device {camera_device_path}")
            exit()

        self.cap.set(cv.CAP_PROP_FOURCC, codec)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv.CAP_PROP_FPS, fps)                
        
    def startVideoThread(self):
        self.running = True
        self.videoThread = threading.Thread(target=self._videoLoop, daemon=True)
        self.videoThread.start()
    
    def stopVideoThread(self):
        self.running = False
        if self.videoThread:
            self.videoThread.join()
    
    def _videoLoop(self):
        print("Video loop started.")
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame from stream. Receiver might have disconnected.")
                self.running = False
                break
              
            hough_lines = self.get_HoughsLinesP(frame)
            results = self.consolidate_into_two_lines_filtered(hough_lines)
            
            frame_with_lines = self.draw_lines_on_frame(frame, results)
            
            with self.thread_lock: 
                self.latest_results = results
                self.latest_image = frame_with_lines
            
    def getLatestResult(self): 
        with self.thread_lock: return self.latest_results
    
    def getLatestImage(self):
        with self.thread_lock: return self.latest_image
    
    def draw_lines_on_frame(self, frame, lines):
        frame_copy = frame.copy()
        
        if lines is not None and len(lines) > 0:
            for i, line in enumerate(lines):
                if len(line) >= 4:
                    x1, y1, x2, y2 = line[:4]
                    color = (0, 255, 0) if i == 0 else (255, 0, 0)
                    cv.line(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                    cv.circle(frame_copy, (int(x1), int(y1)), 8, (0, 255, 255), -1)
                    cv.circle(frame_copy, (int(x2), int(y2)), 8, (0, 255, 255), -1)
                    text = f"Line {i+1}"
                    text_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2) - 10)
                    cv.putText(frame_copy, text, text_pos, cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        info_text = f"Lines detected: {len(lines) if lines else 0}"
        cv.putText(frame_copy, info_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return frame_copy
    
    def consolidate_into_two_lines_filtered(self, lines, n_iterations=10, theta_weight=100, outlier_threshold_std=2.0):
        if lines is None: return []
        segments = lines.reshape(-1, 4)
        line_params = []
        for x1, y1, x2, y2 in segments:
            if x1 == x2 and y1 == y2: continue
            angle = np.arctan2(y2 - y1, x2 - x1)
            theta = angle + np.pi / 2
            r = x1 * np.cos(theta) + y1 * np.sin(theta)
            if r < 0: r, theta = -r, (theta + np.pi) % (2 * np.pi)
            line_params.append([r, theta % np.pi])
        if len(line_params) < 2: return []
        X = np.array(line_params)
        X[:, 1] *= theta_weight
        initial_indices = np.random.choice(len(X), 2, replace=False)
        centroids = X[initial_indices]
        for _ in range(n_iterations):
            distances_to_centroids = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances_to_centroids, axis=0)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(2)])
            if np.allclose(centroids, new_centroids): break
            centroids = new_centroids
        point_distances = np.array([distances_to_centroids[labels[i], i] for i in range(len(X))])
        distance_mean, distance_std = point_distances.mean(), point_distances.std()
        distance_threshold = distance_mean + outlier_threshold_std * distance_std
        filtered_labels = np.where(point_distances <= distance_threshold, labels, -1)
        final_lines = []
        for i in range(2):
            cluster_indices = np.where(filtered_labels == i)[0]
            if len(cluster_indices) < 2: continue
            cluster_points = [p for index in cluster_indices for p in [(segments[index][0], segments[index][1]), (segments[index][2], segments[index][3])]]
            points_array = np.array(cluster_points)
            mean_point = points_array.mean(axis=0)
            centered_data = points_array - mean_point
            covariance_matrix = np.cov(centered_data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            direction_vector = eigenvectors[:, np.argmax(eigenvalues)]
            projected_dist = np.dot(centered_data, direction_vector)
            endpoint1 = mean_point + np.min(projected_dist) * direction_vector
            endpoint2 = mean_point + np.max(projected_dist) * direction_vector
            final_lines.append([int(p) for p in endpoint1] + [int(p) for p in endpoint2])
        return final_lines

    def get_HoughsLinesP(self, image):
        if image is None: return None
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(grayscale, 245, 255, cv.THRESH_BINARY)
        filtered = cv.bitwise_and(grayscale, grayscale, mask=mask)
        dst = cv.Canny(filtered, 50, 150, None, apertureSize=7)
        return cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 10)