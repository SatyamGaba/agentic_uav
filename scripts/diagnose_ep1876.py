import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, numpy as np, math, habitat_sim
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, normalize_to_pixel

dev_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"
with open(dev_path) as f:
    ep = json.load(f)["episodes"][3] # Ep 1876 is index 3

print("Episode ID:", ep["episode_id"])
emb = HabitatGroundEmbodiment(scene_id=resolve_scene_path(ep["scene_id"]), gpu_device_id=-1)
obs = emb.reset(ep)

# Move 1 step forward as in trace:
# Step 1: view=0, u=0.54, v=0.95 -> executed
# Let's inspect step 1 execution
obs = emb.step_discrete(turn_steps_15deg=0, forward_steps_025m=7)
print("After step 1, pos:", emb.sim.get_agent(0).get_state().position)
camera_cfg = CameraConfig()
depth_img = obs.depth_views[0]

for v in [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60]:
    x0, y0 = normalize_to_pixel(0.59, v, camera_cfg)
    d = float(depth_img[y0, x0])
    print(f"v={v:.2f} (y0={y0}): raw_depth={d:.3f}m")

emb.close()

