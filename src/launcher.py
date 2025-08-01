#!/usr/bin/env python3
"""
Launcher script with options to run different components
"""
import sys
import asyncio
import threading
from detector import Detector
from tracker import Tracker
from line_viewer import LineViewer

def run_line_viewer_only():
    """Run just the line viewer"""
    print("Starting line viewer only...")
    viewer = LineViewer()
    viewer.start()

def run_with_viewer():
    """Run main.py with line viewer"""
    from main import run
    
    # Start detector and tracker
    detector = Detector()
    detector.startVideoThread()
    tracker = Tracker(detector)
    tracker.startTrackerThread()
    
    # Start line viewer in separate thread
    line_viewer = LineViewer()
    line_viewer.detector = detector  # Share the same detector
    viewer_thread = threading.Thread(target=line_viewer.start, daemon=True)
    viewer_thread.start()
    
    # Run main drone control
    asyncio.run(run())

def show_help():
    print("Usage:")
    print("  python launcher.py viewer      - Run line viewer only")
    print("  python launcher.py main        - Run main drone control without viewer")
    print("  python launcher.py both        - Run both drone control and viewer")
    print("  python launcher.py help        - Show this help")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        show_help()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "viewer":
        run_line_viewer_only()
    elif command == "main":
        from main import main
        main()
    elif command == "both":
        run_with_viewer()
    elif command == "help":
        show_help()
    else:
        print(f"Unknown command: {command}")
        show_help()
        sys.exit(1)
