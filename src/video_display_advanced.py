#!/usr/bin/env python3
"""
Advanced video display script showing original frame, edge detection, and final result.
"""

import cv2 as cv
import numpy as np
import time
from detector import Detector

class VideoDisplayAdvanced:
    def __init__(self):
        self.detector = Detector()
        self.show_edges = True
        self.show_hough_lines = False
        
    def get_edge_detection_frame(self, frame):
        """Get the edge detection visualization."""
        if frame is None:
            return None
            
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(grayscale, 245, 255, cv.THRESH_BINARY)
        filtered = cv.bitwise_and(grayscale, grayscale, mask=mask)
        edges = cv.Canny(filtered, 50, 150, apertureSize=7)
        return cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    
    def get_hough_lines_frame(self, frame):
        """Get frame with all Hough lines drawn."""
        if frame is None:
            return None
            
        frame_copy = frame.copy()
        hough_lines = self.detector.get_HoughsLinesP(frame)
        
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                cv.line(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 1)  # Red thin lines
        
        # Add info
        count = len(hough_lines) if hough_lines is not None else 0
        cv.putText(frame_copy, f"Raw Hough Lines: {count}", (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame_copy
    
    def create_display_grid(self, original_frame):
        """Create a 2x2 grid showing different processing stages."""
        if original_frame is None:
            return None
            
        # Get processed frame with final detections
        processed_frame = self.detector.getLatestImage()
        if processed_frame is None:
            processed_frame = original_frame.copy()
        
        # Get edge detection frame
        edges_frame = self.get_edge_detection_frame(original_frame)
        
        # Get Hough lines frame
        hough_frame = self.get_hough_lines_frame(original_frame)
        
        # Resize all frames to same size
        height, width = 300, 400
        original_resized = cv.resize(original_frame, (width, height))
        processed_resized = cv.resize(processed_frame, (width, height))
        edges_resized = cv.resize(edges_frame, (width, height))
        hough_resized = cv.resize(hough_frame, (width, height))
        
        # Add labels
        cv.putText(original_resized, "Original", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(processed_resized, "Final Detection", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(edges_resized, "Edge Detection", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(hough_resized, "Raw Hough Lines", (10, 25), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create 2x2 grid
        top_row = np.hstack((original_resized, processed_resized))
        bottom_row = np.hstack((edges_resized, hough_resized))
        grid = np.vstack((top_row, bottom_row))
        
        return grid
    
    def run(self):
        print("Starting advanced video display...")
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  '1' - Show simple detection view")
        print("  '2' - Show 2x2 grid view")
        
        # Start detector
        self.detector.startVideoThread()
        
        show_grid = True
        
        try:
            while True:
                if show_grid:
                    # Get original frame directly from camera
                    ret, original_frame = self.detector.cap.read()
                    if ret:
                        display_frame = self.create_display_grid(original_frame)
                        window_name = 'MAV Ranch House - Processing Pipeline'
                    else:
                        continue
                else:
                    # Show simple detection view
                    display_frame = self.detector.getLatestImage()
                    window_name = 'MAV Ranch House - Line Detection'
                
                if display_frame is not None:
                    cv.imshow(window_name, display_frame)
                
                # Handle key presses
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    timestamp = int(time.time())
                    filename = f"detection_advanced_{timestamp}.jpg"
                    if display_frame is not None:
                        cv.imwrite(filename, display_frame)
                        print(f"Screenshot saved as {filename}")
                elif key == ord('1'):
                    show_grid = False
                    cv.destroyAllWindows()
                    print("Switched to simple detection view")
                elif key == ord('2'):
                    show_grid = True
                    cv.destroyAllWindows()
                    print("Switched to 2x2 grid view")
                
                time.sleep(0.033)  # ~30 FPS
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.detector.stopVideoThread()
            cv.destroyAllWindows()
            print("Done!")

def main():
    display = VideoDisplayAdvanced()
    display.run()

if __name__ == "__main__":
    main()
