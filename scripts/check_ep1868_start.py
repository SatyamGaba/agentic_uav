import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "1868")
camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
agent_pos = emb.sim.get_agent(0).get_state().position
agent_rot = emb.sim.get_agent(0).get_state().rotation
yaw = 2.0 * math.atan2(agent_rot.y, agent_rot.w)
print("Ep 1868 Start pos:", agent_pos, "yaw:", yaw)
print("Goal pos:", ep["goals"][0]["position"])

# Check path
pf = emb.sim.pathfinder
path = habitat_sim.ShortestPath()
path.requested_start = agent_pos
path.requested_end = np.array(ep["goals"][0]["position"], dtype=np.float32)
found = pf.find_path(path)
print("Shortest path found:", found, "length:", path.geodesic_distance)
for i, pt in enumerate(path.points[:5]):
    print(f"  wp {i}: {pt}")

# Check depth in all 4 views
for v_idx in range(4):
    d = obs.depth_views[v_idx]
    print(f"View {v_idx}: min={d.min():.3f}, max={d.max():.3f}, mean={d.mean():.3f}")

emb.close()

