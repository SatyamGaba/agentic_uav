import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"])
sim = emb.sim
pf = sim.pathfinder

pos = np.array([-1.70, 3.3189, 1.95], dtype=np.float32)
goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)

path = habitat_sim.ShortestPath()
path.requested_start = pos
path.requested_end = goal_pos
pf.find_path(path)
pts = list(path.points)

print(f"Total waypoints: {len(pts)}")
for i in range(1, len(pts)):
    pt = pts[i]
    p_test = habitat_sim.ShortestPath()
    p_test.requested_start = pos
    p_test.requested_end = pt
    pf.find_path(p_test)
    is_straight = (len(p_test.points) == 2)
    euclid_d = float(np.linalg.norm(pt - pos))
    geod_d = p_test.geodesic_distance
    print(f"  wp {i}: dist={euclid_d:.2f}m, geod={geod_d:.2f}m, num_pts={len(p_test.points)}, straight={is_straight}")

emb.close()

