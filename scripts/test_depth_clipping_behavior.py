import habitat_sim
import numpy as np

# Create a simple test with habitat-sim
# Let's inspect what depth sensor renders when an object is closer than d_min
cfg = habitat_sim.SimulatorConfiguration()
cfg.scene_id = "agenticnav_paper_replica/data/test_scene.glb" if False else "/home/krutik/data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb"

sim_cfg = habitat_sim.Configuration(cfg, [])
sim = habitat_sim.Simulator(sim_cfg)

# Add depth sensor with d_min = 0.50
camera_spec = habitat_sim.CameraSensorSpec()
camera_spec.uuid = "depth_sensor"
camera_spec.sensor_type = habitat_sim.SensorType.DEPTH
camera_spec.resolution = [480, 640]
camera_spec.position = [0.0, 1.25, 0.0]
camera_spec.hfov = 90.0
# Note: Habitat depth sensor has near/far clipping?
# Let's check attributes of CameraSensorSpec
print("Sensor spec attributes:", [attr for attr in dir(camera_spec) if not attr.startswith("_")])

sim.close()

