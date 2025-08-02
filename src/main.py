
import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, VelocityBodyYawspeed)
from detector import Detector
from tracker import Tracker
from line_viewer import LineViewer
import time
import threading

detector = None
tracker = None
drone = None
line_viewer = None

async def run():
    drone = System()
    # await drone.connect(system_address="udpin://0.0.0.0:14551")
    await drone.connect(system_address="serial:///dev/ttyAMA0:921600")

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
        print(f"Starting offboard mode failed with error code: {error._result.result}")
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

    print("--starting manual control")
    while True:
        goal = tracker.getLatestControl()
        print(goal)
        await drone.offboard.set_velocity_body(goal)
        await asyncio.sleep(0.1)

if __name__ == '__main__':
    detector = Detector()
    detector.startVideoThread()
    tracker = Tracker(detector)
    tracker.startTrackerThread()
    
    # Start line viewer in a separate thread (optional - comment out if not needed)
    line_viewer = LineViewer()
    line_viewer.detector = detector  # Share the same detector instance
    viewer_thread = threading.Thread(target=line_viewer.start, daemon=True)
    viewer_thread.start()
    
    # Run the main drone control loop
    asyncio.run(run())
