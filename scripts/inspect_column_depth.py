import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, normalize_to_pixel

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
obs = emb.step_discrete(2, 8)

# Check depth along column x0 for view 1 at u=0.18
x0 = int(round(0.18 * 640))
col = obs.depth_views[1][:, x0]
print(f"Column {x0} depths:")
for y in range(0, 480, 40):
    print(f"  y={y} (v={y/480:.2f}): depth={col[y]:.3f}m")
emb.close()

