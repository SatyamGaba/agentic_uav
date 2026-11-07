import json
import numpy as np
import habitat_sim
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
scene_path = resolve_scene_path(ep["scene_id"])
config = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=scene_path, camera_config=config)
obs = emb.reset(ep)

# Replay steps 1, 2, 3
obs = emb.step_discrete(2, 8)
obs = emb.step_discrete(-4, 7)
obs = emb.step_discrete(-3, 9)

raw = emb.sim.get_sensor_observations()
d0_raw = raw["depth_view_0"]
print("Raw min depth:", np.min(d0_raw), "max:", np.max(d0_raw))
print("Pixels < 0.05:", np.sum(d0_raw < 0.05))
print("Pixels == 0.0:", np.sum(d0_raw == 0.0))
print("Pixels between 0.0 and 0.10:", np.sum((d0_raw >= 0.0) & (d0_raw < 0.10)))

# Agent state
state = emb.sim.get_agent(0).get_state()
print("Agent pos:", state.position)
print("Agent rot:", state.rotation)
pf = emb.sim.pathfinder
print("Pathfinder is_navigable(pos):", pf.is_navigable(state.position))
print("Distance to closest obstacle on navmesh:", pf.distance_to_closest_obstacle(state.position))

# Where are the min depth pixels in View 0?
min_idx = np.unravel_index(np.argmin(d0_raw), d0_raw.shape)
print("Min pixel row, col:", min_idx, "value:", d0_raw[min_idx])
row, col = min_idx
z_val = d0_raw[row, col]
y_c = (row - config.cy) * z_val / config.fy
x_c = (col - config.cx) * z_val / config.fx
y_agent = config.camera_height - y_c
print(f"In camera frame: x={x_c:.4f}, y={y_c:.4f}, z={z_val:.4f}")
print(f"In agent frame: height above floor datum Y={y_agent:.4f}m")

# Also check around the minimum
sorted_indices = np.argsort(d0_raw.flatten())
for i in range(5):
    idx = np.unravel_index(sorted_indices[i], d0_raw.shape)
    r, c = idx
    val = d0_raw[r, c]
    yc = (r - config.cy) * val / config.fy
    ya = config.camera_height - yc
    print(f"  Rank {i}: row={r}, col={c}, depth={val:.4f}m, height Y={ya:.4f}m")

emb.close()
