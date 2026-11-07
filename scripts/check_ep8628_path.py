import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig
from agenticnav_paper_replica.harness.runner import run_episode

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

config = ReplicaConfig()
config.max_simulator_actions = 500
config.max_decision_steps = 25
config.robot_radius = 0.10
config.obstacle_point_threshold = 5

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)

# Let's inspect the shortest path points in Ep 8628
agent_state = emb.sim.get_agent(0).get_state()
pos = agent_state.position
yaw = 2.0 * math.atan2(agent_state.rotation.y, agent_state.rotation.w)
goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)

path = habitat_sim.ShortestPath()
path.requested_start = pos
path.requested_end = goal_pos
emb.sim.pathfinder.find_path(path)
print("Ep 8628 Start pos:", pos, "yaw:", yaw)
print("Goal pos:", goal_pos)
print("Geodesic distance:", path.geodesic_distance)
print(f"Total waypoints: {len(path.points)}")
for i, pt in enumerate(path.points[:10]):
    print(f"  wp {i}: {pt}")

emb.close()

