import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"])
sim = emb.sim
pf = sim.pathfinder

pos = np.array([-1.70036, 3.3189, 1.95108], dtype=np.float32)
wp1 = np.array([-1.84825, 3.3189, 1.90178], dtype=np.float32)

vec = wp1 - pos
d = np.linalg.norm(vec)
dir_v = vec / d

# Try stepping 0.15m, 0.25m, 0.35m along dir_v
for step_d in [0.15, 0.25, 0.35]:
    cand = pos + dir_v * step_d
    is_nav = pf.is_navigable(cand)
    snap = pf.snap_point(cand)
    snap_err = np.linalg.norm(snap - cand)
    print(f"Step {step_d:.2f}m: navigable={is_nav}, snap_err={snap_err:.3f}")

emb.close()

