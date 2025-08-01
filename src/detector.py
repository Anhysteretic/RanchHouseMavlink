import cv2 as cv
import numpy as np
import threading
import time

class Detector():
    
    def __init__(self):        
        self.latest_image = None
        self.latest_results = np.array([[0, 0, 0, 0], [0, 0, 0, 0]])
        self.thread_lock = threading.Lock()
        
        self.running = False
        self.videoThread = None
                
        self.cap = cv.VideoCapture(self.gstreamer_pipeline(), cv.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print("Failed to open GStreamer pipeline.")
            exit()
        print("GStreamer pipeline opened successfully. Press 'q' to quit.")
        
    def startVideoThread(self):
        self.running = True
        print("[DEBUG] Starting video detection thread...")
        self.videoThread = threading.Thread(target=self._videoLoop, daemon=True)
        self.videoThread.start()
        print("[DEBUG] Video thread started successfully")
    
    def stopVideoThread(self):
        if self.running:
            print("[DEBUG] Stopping video thread...")
            self.running = False
            self.videoThread.join()
            print("[DEBUG] Video thread stopped successfully")
        else:
            print("[DEBUG] Video thread was not running")
        
    def gstreamer_pipeline(self):
        """Defines the GStreamer pipeline for receiving and decoding the H.264 stream."""
        return (
            "udpsrc port=5000 ! "
            "application/x-rtp, media=(string)video, clock-rate=(int)90000, encoding-name=(string)H264 ! "
            "rtph264depay ! "
            "avdec_h264 ! "
            "videoconvert ! "
            "appsink"
        )
    
    def _videoLoop(self):
        frame_count = 0
        loop_start_time = time.time()
        print("[DEBUG] Video loop started")
        
        while self.running:
            frame_start_time = time.time()
            ret, frame = self.cap.read()
            if not ret:
                print("[ERROR] Failed to receive frame.")
                return
            
            frame_count += 1
            
            with self.thread_lock:
                self.latest_image = frame
                
                # Time the line detection process
                detection_start = time.time()
                hough_lines = self.get_HoughsLinesP(frame)
                hough_time = time.time() - detection_start
                
                consolidation_start = time.time()
                results = self.consolidate_into_two_lines_filtered(hough_lines)
                consolidation_time = time.time() - consolidation_start
                
                self.latest_results = results
                
                frame_processing_time = time.time() - frame_start_time
                
                # Print debug info every 30 frames (roughly every second at 30fps)
                if frame_count % 30 == 0:
                    total_runtime = time.time() - loop_start_time
                    avg_fps = frame_count / total_runtime
                    print(f"[DEBUG] Frame {frame_count}: "
                          f"Hough: {hough_time*1000:.1f}ms, "
                          f"Consolidation: {consolidation_time*1000:.1f}ms, "
                          f"Total: {frame_processing_time*1000:.1f}ms, "
                          f"Avg FPS: {avg_fps:.1f}, "
                          f"Lines found: {len(results) if results else 0}")
                    
                    if results:
                        print(f"[DEBUG] Line results: {results}")
        
        total_runtime = time.time() - loop_start_time
        print(f"[DEBUG] Video loop finished. Total runtime: {total_runtime:.2f}s, "
              f"Processed {frame_count} frames, Avg FPS: {frame_count/total_runtime:.1f}")
            
    def getLatestResult(self): 
        with self.thread_lock: return self.latest_results
    
    def getLatesImage(self):
        with self.thread_lock: return self.latest_image
    
    def consolidate_into_two_lines_filtered(self, lines, n_iterations=10, theta_weight=100, outlier_threshold_std=2.0):
        """
        Consolidates line segments into a maximum of two lines, with outlier rejection.
        Implemented using only NumPy.

        Args:
            lines (np.ndarray): NumPy array of shape (N, 1, 4) for segments [x1, y1, x2, y2].
            n_iterations (int): Number of iterations for the K-Means algorithm.
            theta_weight (int): Weight to make angular distance comparable to pixel distance.
            outlier_threshold_std (float): How many standard deviations from the mean distance
                                        to use as the outlier threshold. Lower is stricter.

        Returns:
            list: A list containing up to two consolidated line segments.
        """
        
        if lines is None: 
            print("[DEBUG] No Hough lines detected")
            return []

        segments = lines.reshape(-1, 4)
        print(f"[DEBUG] Processing {len(segments)} line segments from Hough transform")
        line_params = []

        # 1. Represent each segment in (r, θ) space
        for x1, y1, x2, y2 in segments:
            if x1 == x2 and y1 == y2: continue
            angle = np.arctan2(y2 - y1, x2 - x1)
            theta = angle + np.pi / 2
            r = x1 * np.cos(theta) + y1 * np.sin(theta)
            if r < 0:
                r, theta = -r, (theta + np.pi) % (2 * np.pi)
            line_params.append([r, theta % np.pi])

        if len(line_params) < 2: 
            # print(f"[DEBUG] Insufficient line parameters: {len(line_params)} (need at least 2)")
            return []

        X = np.array(line_params)
        X[:, 1] *= theta_weight
        print(f"[DEBUG] Created {len(X)} line parameters for clustering")

        # 2. K-Means implementation with NumPy
        initial_indices = np.random.choice(len(X), 2, replace=False)
        centroids = X[initial_indices]
        for _ in range(n_iterations):
            distances_to_centroids = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances_to_centroids, axis=0)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(2)])
            if np.allclose(centroids, new_centroids): break
            centroids = new_centroids

        # 3. Filter outliers based on distance to centroid
        point_distances = np.array([distances_to_centroids[labels[i], i] for i in range(len(X))])
        distance_mean = point_distances.mean()
        distance_std = point_distances.std()
        distance_threshold = distance_mean + outlier_threshold_std * distance_std

        # Create a new set of labels, marking outliers with -1
        filtered_labels = np.where(point_distances <= distance_threshold, labels, -1)

        # 4. Merge non-outlier segments for each cluster
        final_lines = []
        print(f"[DEBUG] Processing clusters after outlier filtering...")
        
        for i in range(2):
            cluster_indices = np.where(filtered_labels == i)[0]
            print(f"[DEBUG] Cluster {i}: {len(cluster_indices)} segments")
            if len(cluster_indices) < 2: continue # Need at least 2 segments to form a line
                
            cluster_points = []
            for index in cluster_indices:
                x1, y1, x2, y2 = segments[index]
                cluster_points.extend([(x1, y1), (x2, y2)])
            
            points_array = np.array(cluster_points)
            
            # PCA with NumPy to find the best-fit line
            mean_point = points_array.mean(axis=0)
            centered_data = points_array - mean_point
            covariance_matrix = np.cov(centered_data, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
            direction_vector = eigenvectors[:, np.argmax(eigenvalues)]
            
            # Project points to find endpoints
            projected_dist = np.dot(centered_data, direction_vector)
            endpoint1 = mean_point + np.min(projected_dist) * direction_vector
            endpoint2 = mean_point + np.max(projected_dist) * direction_vector
            
            line_result = [int(p) for p in endpoint1] + [int(p) for p in endpoint2]
            final_lines.append(line_result)
            print(f"[DEBUG] Cluster {i} consolidated line: {line_result}")

        print(f"[DEBUG] Final result: {len(final_lines)} consolidated lines")
        return final_lines

    def get_HoughsLinesP(self, image):
        if image is not None: 
            grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else: 
            print("[DEBUG] No image provided to get_HoughsLinesP")
            return
        
        _, mask = cv.threshold(grayscale, 245, 255, cv.THRESH_BINARY)
        filtered = cv.bitwise_and(grayscale, grayscale, mask=mask)
        dst = cv.Canny(filtered, 50, 150, None, apertureSize=7)
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, 50, None, 100, 10)
        
        num_lines = len(linesP) if linesP is not None else 0
        if num_lines == 0:
            print("[DEBUG] No lines detected by HoughLinesP")
        
        return linesP