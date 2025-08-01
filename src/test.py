
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
    
    print("Video display started. Press 'q' in video window to close video (drone will continue).")
    
    while video_display_running:
        try:
            # Get the latest processed frame with detections
            frame = detector.getLatestImage()
            
            if frame is not None:
                # Resize for display if needed
                height, width = frame.shape[:2]
                if width > 800:  # Scale down for reasonable window size
                    scale = 800 / width
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    frame = cv.resize(frame, (new_width, new_height))
                
                # Add drone status overlay
                cv.putText(frame, "MAV Ranch House - Live Detection", (10, height - 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv.putText(frame, "Press 'q' to close video", (10, height - 10), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Display the frame
                cv.imshow('MAV Ranch House - Drone Video Feed', frame)
                
                # Handle key presses
                key = cv.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Video display closed by user")
                    break
            
            time.sleep(0.033)  # ~30 FPS
            
        except Exception as e:
            print(f"Video display error: {e}")
            time.sleep(0.1)
    
    cv.destroyAllWindows()
    video_display_running = False

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
    video_display_running = False
    cv.destroyAllWindows()

async def run():

    drone = System()
    await drone.connect(system_address="udpin://0.0.0.0:14551")

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"-- Connected to drone!")
            break
    
    # print("Waiting for drone to have a global position estimate...")
    # async for health in drone.telemetry.health():
    #     if health.is_global_position_ok and health.is_home_position_ok:
    #         print("-- Global position estimate OK")
    #         break

    print("-- Arming")
    await drone.action.arm()

    await drone.offboard.set_velocity_body(VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    print("-- Starting offboard")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Starting offboard mode failed with error code: \
              {error._result.result}")
        print("-- Disarming")
        await drone.action.disarm()
        return
    
    print("-- Setting initial setpoint")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, -0.5, 0.0))
    await asyncio.sleep(7.5)

    print("-- Wait for a bit")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0))
    await asyncio.sleep(2)

    print("-- Fly a circle")
    await drone.offboard.set_velocity_body(
        VelocityBodyYawspeed(0.0, 0.0, 0.0, 30.0))
    await asyncio.sleep(10)

    print("-- Stopping offboard")
    try:
        await drone.offboard.stop()
    except OffboardError as error:
        print(f"Stopping offboard mode failed with error code: \
              {error._result.result}")
    
    print("-- Disarming")
    await drone.action.disarm()
    
    # Stop video display
    stop_video_display()
    print("Flight test completed!")

if __name__ == '__main__':
    # print("Starting MAV Ranch House test with video display...")
    
    # # Create detector and tracker
    # detector = Detector("192.168.0.102")
    # detector.startVideoThread()
    # tracker = Tracker(detector)
    # tracker.startTrackerThread()
    
    # # Start video display
    # start_video_display(detector)
    
    # try:
        # Run the drone test
        asyncio.run(run())
    # except KeyboardInterrupt:
    #     print("\nTest interrupted by user")
    # finally:
    #     # Clean up
    #     print("Cleaning up...")
    #     stop_video_display()
    #     if detector.running:
    #         detector.stopVideoThread()
    #     if tracker.running:
    #         tracker.stopTrackerThread()
    #     print("Test completed!")
