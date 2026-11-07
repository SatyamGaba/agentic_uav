import airsim
import time

try:
    c = airsim.MultirotorClient(port=41451)
    c.confirmConnection()
    print("1. Calling c.reset()...")
    c.reset()
    time.sleep(1.0)
    print("2. Calling enableApiControl...")
    c.enableApiControl(True, vehicle_name='drone_1')
    c.armDisarm(True, vehicle_name='drone_1')
    print("Reset test SUCCESS!")
except Exception as e:
    print(f"Error during reset: {type(e).__name__}: {e}")
