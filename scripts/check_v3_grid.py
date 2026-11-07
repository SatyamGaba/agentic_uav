import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "1868")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

d3 = obs.depth_views[3]
print("View 3 depth shape:", d3.shape)
# Print a 10x10 downsampled grid of depths in View 3
grid = d3[::48, ::48]
print("View 3 depth grid (rows = top to bottom, cols = left to right):")
for r in range(grid.shape[0]):
    row_str = " ".join([f"{grid[r, c]:4.2f}" for c in range(grid.shape[1])])
    print(f"y={r*48:3d}: {row_str}")

emb.close()

