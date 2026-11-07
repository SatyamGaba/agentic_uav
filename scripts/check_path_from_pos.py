import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
sim = emb.sim
pos = np.array([-1.78, 3.3189, 3.82], dtype=np.float32)
goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)

path = habitat_sim.ShortestPath()
path.requested_start = pos
path.requested_end = goal_pos
sim.pathfinder.find_path(path)
print("From [-1.78, 3.82] to goal:")
print("Found:", len(path.points) > 0, "Length:", path.geodesic_distance)
for i, pt in enumerate(path.points[:8]):
    d = np.linalg.norm(pt - pos)
    print(f"  wp {i}: {pt}, dist={d:.2f}m")

emb.close()

