import sys
from pathlib import Path
import habitat_sim
import numpy as np

scene_path = "/home/krutik/data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb"
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = scene_path
sim_cfg.gpu_device_id = -1

backend_cfg = habitat_sim.SimulatorConfiguration()
backend_cfg.scene_id = scene_path
backend_cfg.gpu_device_id = -1

agent_cfg = habitat_sim.agent.AgentConfiguration()
cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
pf = sim.pathfinder

pt_0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
snapped = pf.snap_point(pt_0)
is_nav = pf.is_navigable(snapped)

goal_0 = np.array([-1.7, 0.0, -7.5], dtype=np.float32)
goal_snapped = pf.snap_point(goal_0)

path = habitat_sim.ShortestPath()
path.requested_start = snapped
path.requested_end = goal_snapped
found = pf.find_path(path)

print(f"Start snapped: {snapped}, is_nav: {is_nav}")
print(f"Goal snapped: {goal_snapped}, is_nav: {pf.is_navigable(goal_snapped)}")
print(f"Path found: {found}, distance: {path.geodesic_distance if found else None}")

sim.close()

