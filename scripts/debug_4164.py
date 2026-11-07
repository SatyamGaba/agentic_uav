import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points, normalize_to_pixel, convert_raw_depth, deproject_pixel, camera_to_agent_frame
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "4164")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

# Execute steps 1 to 4:
# We can run the same smart oracle for 4 steps and inspect step 5
# Or print the trajectory
from scripts.test_smart_oracle_v2 import test_smart_oracle
# Let's inspect step 5 by stepping

