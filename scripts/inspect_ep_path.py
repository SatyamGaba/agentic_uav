import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, numpy as np, habitat_sim
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path

dev_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"
with open(dev_path) as f:
    data = json.load(f)
    ep = data.get("episodes", data)[0]
emb = HabitatGroundEmbodiment(scene_id=resolve_scene_path(ep["scene_id"]), gpu_device_id=-1)
obs = emb.reset(ep)
sim = emb.sim
pf = sim.pathfinder
path = habitat_sim.ShortestPath()
path.requested_start = sim.get_agent(0).get_state().position
path.requested_end = np.array(ep["goals"][0]["position"], dtype=np.float32)
pf.find_path(path)
print("Path found:", len(path.points))
for i, pt in enumerate(path.points):
    d = float(np.linalg.norm(pt - path.points[i-1])) if i > 0 else 0.0
    print(f"pt {i}: {pt}, dist from prev: {d:.2f}m")
emb.close()
