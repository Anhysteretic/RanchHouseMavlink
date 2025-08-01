import tkinter as tk
from tkinter import ttk
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import threading
import time

class VideoStreamGUI:
    def __init__(self, detector):
        self.detector = detector
        self.root = tk.Tk()
        self.root.title("MAV Ranch House - Video Stream")
        self.root.geometry("800x600")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        self.main_frame.columnconfigure(1, weight=1)
        self.main_frame.rowconfigure(0, weight=1)
        
        # Create video display label
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        
        # Create control panel
        self.create_control_panel()
        
        # Initialize video display variables
        self.display_width = 640
        self.display_height = 480
        self.running = False
        self.update_thread = None
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def create_control_panel(self):
        """Create the control panel with buttons and information display."""
        control_frame = ttk.Frame(self.main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Title
        title_label = ttk.Label(control_frame, text="Video Stream Control", font=("Arial", 12, "bold"))
        title_label.grid(row=0, column=0, sticky=tk.W, pady=(0, 10))
        
        # Start/Stop buttons
        self.start_button = ttk.Button(control_frame, text="Start Stream", command=self.start_stream)
        self.start_button.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=2)
        
        self.stop_button = ttk.Button(control_frame, text="Stop Stream", command=self.stop_stream, state="disabled")
        self.stop_button.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=2)
        
        # Separator
        separator = ttk.Separator(control_frame, orient='horizontal')
        separator.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Detection info
        info_label = ttk.Label(control_frame, text="Detection Info", font=("Arial", 10, "bold"))
        info_label.grid(row=4, column=0, sticky=tk.W, pady=(0, 5))
        
        # Status display
        self.status_var = tk.StringVar(value="Stream: Stopped")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=5, column=0, sticky=tk.W, pady=2)
        
        # Lines detected display
        self.lines_var = tk.StringVar(value="Lines detected: 0")
        lines_label = ttk.Label(control_frame, textvariable=self.lines_var)
        lines_label.grid(row=6, column=0, sticky=tk.W, pady=2)
        
        # FPS display
        self.fps_var = tk.StringVar(value="FPS: 0")
        fps_label = ttk.Label(control_frame, textvariable=self.fps_var)
        fps_label.grid(row=7, column=0, sticky=tk.W, pady=2)
        
        # Separator
        separator2 = ttk.Separator(control_frame, orient='horizontal')
        separator2.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=10)
        
        # Display options
        options_label = ttk.Label(control_frame, text="Display Options", font=("Arial", 10, "bold"))
        options_label.grid(row=9, column=0, sticky=tk.W, pady=(0, 5))
        
        self.show_lines_var = tk.BooleanVar(value=True)
        show_lines_check = ttk.Checkbutton(control_frame, text="Show detected lines", variable=self.show_lines_var)
        show_lines_check.grid(row=10, column=0, sticky=tk.W, pady=2)
        
        self.show_raw_var = tk.BooleanVar(value=False)
        show_raw_check = ttk.Checkbutton(control_frame, text="Show raw Canny edges", variable=self.show_raw_var)
        show_raw_check.grid(row=11, column=0, sticky=tk.W, pady=2)
        
    def start_stream(self):
        """Start the video stream display."""
        if not self.running:
            self.running = True
            self.start_button.config(state="disabled")
            self.stop_button.config(state="normal")
            self.status_var.set("Stream: Starting...")
            
            # Start the detector video thread if not already running
            if not self.detector.running:
                self.detector.startVideoThread()
            
            # Start GUI update thread
            self.update_thread = threading.Thread(target=self.update_display, daemon=True)
            self.update_thread.start()
            
    def stop_stream(self):
        """Stop the video stream display."""
        if self.running:
            self.running = False
            self.start_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.status_var.set("Stream: Stopped")
            self.fps_var.set("FPS: 0")
            self.lines_var.set("Lines detected: 0")
            
    def update_display(self):
        """Update the video display in a separate thread."""
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            try:
                # Get the latest frame from detector
                ret, frame = self.detector.cap.read()
                if not ret:
                    time.sleep(0.033)  # ~30 FPS
                    continue
                
                # Store the latest image in detector for other uses
                with self.detector.thread_lock:
                    self.detector.latest_image = frame.copy()
                
                # Process frame for display
                display_frame = self.process_frame_for_display(frame)
                
                # Convert to PIL Image and then to PhotoImage
                display_frame_rgb = cv.cvtColor(display_frame, cv.COLOR_BGR2RGB)
                pil_image = Image.fromarray(display_frame_rgb)
                
                # Resize to fit display
                pil_image = pil_image.resize((self.display_width, self.display_height), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update GUI in main thread
                self.root.after(0, self.update_video_label, photo)
                
                # Update FPS
                frame_count += 1
                if frame_count % 30 == 0:  # Update every 30 frames
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time
                    self.root.after(0, self.fps_var.set, f"FPS: {fps:.1f}")
                
                # Update detection info
                latest_results = self.detector.getLatestResult()
                num_lines = len(latest_results) if latest_results is not None else 0
                self.root.after(0, self.lines_var.set, f"Lines detected: {num_lines}")
                self.root.after(0, self.status_var.set, "Stream: Running")
                
                time.sleep(0.033)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in display update: {e}")
                time.sleep(0.1)
                
    def process_frame_for_display(self, frame):
        """Process the frame for display with overlays."""
        display_frame = frame.copy()
        
        if self.show_raw_var.get():
            # Show Canny edge detection
            grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, mask = cv.threshold(grayscale, 245, 255, cv.THRESH_BINARY)
            filtered = cv.bitwise_and(grayscale, grayscale, mask=mask)
            edges = cv.Canny(filtered, 50, 150, apertureSize=7)
            display_frame = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
        
        if self.show_lines_var.get():
            # Draw detected lines (they should already be drawn by detector)
            latest_results = self.detector.getLatestResult()
            if latest_results is not None and len(latest_results) > 0:
                for i, line in enumerate(latest_results):
                    if len(line) >= 4:
                        x1, y1, x2, y2 = line[:4]
                        # Use different colors for different lines
                        color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Green for first, Blue for second
                        cv.line(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        # Draw endpoints
                        cv.circle(display_frame, (int(x1), int(y1)), 4, (0, 255, 255), -1)
                        cv.circle(display_frame, (int(x2), int(y2)), 4, (0, 255, 255), -1)
        
        return display_frame
        
    def update_video_label(self, photo):
        """Update the video label with new photo (called from main thread)."""
        self.video_label.configure(image=photo)
        self.video_label.image = photo  # Keep a reference
        
    def on_closing(self):
        """Handle window closing event."""
        self.stop_stream()
        if self.detector.running:
            self.detector.stopVideoThread()
        self.root.destroy()
        
    def run(self):
        """Start the GUI main loop."""
        self.root.mainloop()

def main():
    """Main function to run the GUI with detector."""
    from detector import Detector
    
    # Create detector instance
    detector = Detector()
    
    # Create and run GUI
    gui = VideoStreamGUI(detector)
    gui.run()

if __name__ == "__main__":
    main()
