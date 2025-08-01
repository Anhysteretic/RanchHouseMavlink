#!/usr/bin/env python3
"""
Test video display with webcam or test video file for debugging.
"""

import cv2 as cv
import numpy as np
import time

class TestDetector:
    """Simplified detector for testing without GStreamer."""
    
    def __init__(self, use_webcam=True):
        self.latest_image = None
        self.latest_results = []
        self.running = False
        
        if use_webcam:
            # Try to use webcam
            self.cap = cv.VideoCapture(0)  # Default webcam
            if not self.cap.isOpened():
                print("Failed to open webcam, creating test pattern...")
                self.cap = None
        else:
            self.cap = None
            
    def create_test_frame(self):
        """Create a test frame with some lines for testing."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame.fill(50)  # Dark gray background
        
        # Draw some test lines
        cv.line(frame, (100, 100), (500, 400), (255, 255, 255), 5)  # White line
        cv.line(frame, (200, 50), (400, 450), (255, 255, 255), 5)   # Another white line
        
        # Add some noise
        noise = np.random.randint(0, 50, frame.shape, dtype=np.uint8)
        frame = cv.add(frame, noise)
        
        return frame
    
    def get_frame(self):
        """Get a frame from webcam or create test pattern."""
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        
        # Fallback to test pattern
        return self.create_test_frame()
    
    def get_HoughsLinesP(self, image):
        """Simplified Hough line detection."""
        if image is None:
            return None
            
        grayscale = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(grayscale, 200, 255, cv.THRESH_BINARY)  # Lower threshold for test
        edges = cv.Canny(mask, 50, 150, apertureSize=3)
        return cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
    def draw_lines_on_frame(self, frame, lines):
        """Draw detected lines on frame."""
        frame_copy = frame.copy()
        
        if lines is not None and len(lines) > 0:
            for i, line in enumerate(lines):
                x1, y1, x2, y2 = line[0]
                
                # Use different colors for different lines
                color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)  # Green/Blue alternating
                
                # Draw the main line
                cv.line(frame_copy, (x1, y1), (x2, y2), color, 3)
                
                # Draw endpoints
                cv.circle(frame_copy, (x1, y1), 8, (0, 255, 255), -1)
                cv.circle(frame_copy, (x2, y2), 8, (0, 255, 255), -1)
        
        # Add detection info
        count = len(lines) if lines is not None else 0
        cv.putText(frame_copy, f"Lines detected: {count}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv.putText(frame_copy, f"Lines detected: {count}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)  # Black outline
        
        return frame_copy
    
    def process_frame(self):
        """Process a single frame and return result with lines drawn."""
        frame = self.get_frame()
        if frame is None:
            return None
            
        # Detect lines
        hough_lines = self.get_HoughsLinesP(frame)
        
        # Draw lines on frame
        result_frame = self.draw_lines_on_frame(frame, hough_lines)
        
        return result_frame
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap is not None:
            self.cap.release()

def main():
    print("Starting test video display...")
    print("This will try to use your webcam, or create test patterns if no webcam is available")
    print("Press 'q' to quit, 's' to save screenshot")
    
    # Create test detector
    detector = TestDetector(use_webcam=True)  # Set to False to use test pattern only
    
    try:
        while True:
            # Process frame
            frame = detector.process_frame()
            
            if frame is not None:
                # Resize if too large
                height, width = frame.shape[:2]
                if width > 1000:
                    scale = 1000 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv.resize(frame, (new_width, new_height))
                
                # Display frame
                cv.imshow('Test Line Detection', frame)
                
                # Handle key presses
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"test_detection_{timestamp}.jpg"
                    cv.imwrite(filename, frame)
                    print(f"Screenshot saved as {filename}")
            
            time.sleep(0.033)  # ~30 FPS
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        detector.cleanup()
        cv.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main()
