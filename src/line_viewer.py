#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import threading
import time
from detector import Detector

class LineViewer:
    def __init__(self, detector=None):
        self.detector = detector
        self.running = False
        self.window_name = "Line Detection Viewer"
        self.owns_detector = detector is None  # Track if we created the detector
        
    def start(self):
        """Start the detector and display window"""
        print("Starting line viewer...")
        
        # Initialize detector if not provided
        if self.detector is None:
            try:
                self.detector = Detector()
                self.owns_detector = True
            except Exception as e:
                print(f"Error initializing detector: {e}")
                print("Make sure your camera is connected and accessible at /dev/video0")
                return
        
        # Start the detector video thread only if we own it
        if self.owns_detector:
            self.detector.startVideoThread()
        
        self.running = True
        
        # Create window
        cv.namedWindow(self.window_name, cv.WINDOW_AUTOSIZE)
        
        print("Press 'q' to quit, 'ESC' to exit, 's' to save current frame")
        
        frame_count = 0
        try:
            while self.running:
                # Get the latest frame with drawn lines
                frame = self.detector.getLatestImage()
                
                if frame is not None:
                    # Add frame counter and instructions
                    frame_copy = frame.copy()
                    cv.putText(frame_copy, f"Frame: {frame_count}", 
                             (10, frame.shape[0] - 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv.putText(frame_copy, "Press 'q' to quit, 's' to save", 
                             (10, frame.shape[0] - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Display the frame
                    cv.imshow(self.window_name, frame_copy)
                    frame_count += 1
                else:
                    # Show a black frame with text if no image available
                    black_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv.putText(black_frame, "Waiting for camera feed...", 
                             (400, 360), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv.putText(black_frame, "Make sure camera is connected", 
                             (420, 400), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                    cv.imshow(self.window_name, black_frame)
                
                # Check for key press
                key = cv.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:  # 'q' or ESC key
                    break
                elif key == ord('s') and frame is not None:  # 's' to save
                    filename = f"line_detection_frame_{int(time.time())}.jpg"
                    cv.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                    
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the viewer and cleanup"""
        print("Stopping line viewer...")
        self.running = False
        
        # Stop detector only if we own it
        if self.detector and self.owns_detector:
            self.detector.stopVideoThread()
        
        # Cleanup OpenCV
        cv.destroyAllWindows()
        
        print("Line viewer stopped")

def main():
    viewer = LineViewer()
    try:
        viewer.start()
    except Exception as e:
        print(f"Error: {e}")
        viewer.stop()

if __name__ == "__main__":
    main()
