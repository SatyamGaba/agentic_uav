import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
camera_cfg = CameraConfig()
config = ReplicaConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
# Step 1
obs = emb.step_discrete(2, 8)
# Step 2
obs = emb.step_discrete(-4, 7)

# Now check Step 3 before the collision:
# What was the obstacle cloud if we include z >= 0.50?
h, w = obs.depth_views[0].shape
for v_idx in [0, 1]:
    d_img = obs.depth_views[v_idx]
    near_count = np.sum((d_img >= 0.49) & (d_img <= 0.51))
    print(f"View {v_idx} has {near_count} pixels at near clamp (~0.50m)")

emb.close()

