#!/usr/bin/env python3
import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)
from detector import Detector
from tracker import Tracker
import time
import cv2 as cv
import threading

# Global variables for video display
video_display_running = False
video_thread = None

def video_display_loop(detector):
    """Video display loop running in separate thread."""
    global video_display_running
    
    print("Video display loop started. Press 'q' in the video window to close it.")
    
    while video_display_running:
        try:
            frame = detector.getLatestImage()
            
            if frame is not None:
                cv.imshow('MAV Ranch House - Drone Video Feed', frame)
                
                # Handle key presses
                if cv.waitKey(1) & 0xFF == ord('q'):
                    print("Video display closed by user.")
                    video_display_running = False # Signal to stop
                    break
            
            time.sleep(0.03) # ~30 FPS
            
        except Exception as e:
            print(f"Video display error: {e}")
            break
    
    cv.destroyAllWindows()
    video_display_running = False
    print("Video display loop finished.")

def start_video_display(detector):
    """Start video display in separate thread."""
    global video_display_running, video_thread
    
    if not video_display_running:
        video_display_running = True
        video_thread = threading.Thread(target=video_display_loop, args=(detector,), daemon=True)
        video_thread.start()

def stop_video_display():
    """Stop video display."""
    global video_display_running
    if video_display_running:
        print("Stopping video display...")
        video_display_running = False
        # The loop will exit and destroy windows on its own
        if video_thread:
            video_thread.join(timeout=1) # Wait a moment for the thread to finish

async def run():
    """
    Main async function. For video testing, this just waits indefinitely.
    Uncomment the drone logic to run a flight test.
    """
    print("Running in video-only test mode.")
    print("Press Ctrl+C in the terminal to stop the script.")
    
    # This line waits forever, keeping the script alive for video display.
    await asyncio.Future() 
    
    # --- DRONE LOGIC (Currently commented out for video testing) ---
    # drone = System()
    # await drone.connect(system_address="udpin://0.0.0.0:14551")

    # print("Waiting for drone to connect...")
    # async for state in drone.core.connection_state():
    #     if state.is_connected:
    #         print(f"-- Connected to drone!")
    #         break
    
    # print("Waiting for drone to have a global position estimate...")
    # async for health in drone.telemetry.health():
    #     if health.is_global_position_ok and health.is_home_position_ok:
    #         print("-- Global position estimate OK")
    #         break

    # print("-- Arming")
    # await drone.action.arm()

    # await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    # print("-- Starting offboard")
    # try:
    #     await drone.offboard.start()
    # except OffboardError as error:
    #     print(f"Starting offboard mode failed with error code: {error._result.result}")
    #     print("-- Disarming")
    #     await drone.action.disarm()
    #     return
    
    # print("-- Setting initial setpoint")
    # await drone.offboard.set_velocity_body(
    #     VelocityBodyYawspeed(0.0, 0.0, -0.5, 0.0))
    # await asyncio.sleep(7.5)

    # print("-- Wait for a bit")
    # await drone.offboard.set_velocity_body(
    #     VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    # await asyncio.sleep(2)

    # print("-- Fly a circle")
    # await drone.offboard.set_velocity_body(
    #     VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
    # await asyncio.sleep(10)

    # print("-- Stopping offboard")
    # try:
    #     await drone.offboard.stop()
    # except OffboardError as error:
    #     print(f"Stopping offboard mode failed with error code: {error._result.result}")
    
    # print("-- Disarming")
    # await drone.action.disarm()

if __name__ == '__main__':
    print("Starting MAV Ranch House test with video display...")
    
    # --- IMPORTANT ---
    # Make sure this IP address is the correct one for your Raspberry Pi sender
    SENDER_IP = "192.168.0.102" 
    
    detector = None
    tracker = None

    try:
        # Create detector and tracker
        detector = Detector(SENDER_IP)
        detector.startVideoThread()
        
        # Assuming you have a Tracker class defined elsewhere
        tracker = Tracker(detector)
        # tracker.startTrackerThread()
        
        # Start video display
        start_video_display(detector)
        
        # Run the main async function
        asyncio.run(run())

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Clean up
        print("Cleaning up...")
        stop_video_display()
        if detector and detector.running:
            detector.stopVideoThread()
        # if tracker and tracker.running:
        #     tracker.stopTrackerThread()
        print("Test completed!")
