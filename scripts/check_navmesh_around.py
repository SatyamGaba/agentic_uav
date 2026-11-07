import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
obs = emb.step_discrete(2, 8)
agent_pos = emb.sim.get_agent(0).get_state().position
agent_rot = emb.sim.get_agent(0).get_state().rotation
yaw = 2.0 * math.atan2(agent_rot.y, agent_rot.w)
print("Agent pos:", agent_pos, "yaw:", yaw)

pf = emb.sim.pathfinder
print("Is agent_pos navigable?", pf.is_navigable(agent_pos))

# In agent frame, heading -60 deg is to the right
# Let's check shortest path points from agent_pos to goal
path = habitat_sim.ShortestPath()
path.requested_start = agent_pos
path.requested_end = np.array(ep["goals"][0]["position"], dtype=np.float32)
found = pf.find_path(path)
print("Shortest path found:", found, "length:", path.geodesic_distance)
for i, pt in enumerate(path.points[:8]):
    print(f"  wp {i}: {pt}")
emb.close()

