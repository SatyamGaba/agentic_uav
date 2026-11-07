import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, normalize_to_pixel

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
# Step 1
obs = emb.step_discrete(2, 8)
# Step 2
obs = emb.step_discrete(-4, 7)
# Step 3
obs = emb.step_discrete(-3, 9)
print("Pos after Step 3:", obs.position, "actual moved:", obs.actual_distance_moved, "collided:", obs.collided)
d0 = obs.depth_views[0]
x0, y0 = normalize_to_pixel(0.49, 0.95, camera_cfg)
print("Depth at view 0 (", x0, ",", y0, "):", d0[y0, x0])
print("Min depth view 0:", d0.min(), "Max depth view 0:", d0.max())
print("Histogram of view 0 depth:")
hist, bins = np.histogram(d0, bins=10)
for h, b in zip(hist, bins):
    print(f"  {b:.2f}m: {h}")
emb.close()

