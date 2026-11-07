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
obs = emb.step_discrete(2, 8)
obs = emb.step_discrete(-4, 7)
obs = emb.step_discrete(-3, 9)

raw = emb.sim.get_sensor_observations()
d0 = raw["depth_view_0"]
zero_mask = (d0 == 0.0)
print(f"Total zero pixels in View 0: {np.sum(zero_mask)}")

rows, cols = np.where(zero_mask)
print(f"Zero pixels row span: [{np.min(rows)}, {np.max(rows)}], col span: [{np.min(cols)}, {np.max(cols)}]")

# Look at the pixels right next to the zeros
# (where depth transition from 0 to non-zero occurs)
for r, c in zip(rows[:10], cols[:10]):
    # check neighbor at c + 1
    c_next = min(c + 5, 639)
    print(f"Pixel ({r}, {c})=0.0, neighbor ({r}, {c_next})={d0[r, c_next]:.4f}m")

# Check View 3 (Left view) and View 1 (Right view)
for vi in [1, 2, 3]:
    d = raw[f"depth_view_{vi}"]
    print(f"View {vi} zeros count: {np.sum(d == 0.0)}, min non-zero: {np.min(d[d > 0]):.4f}m")

# Check distance to closest obstacle in direction of the agent
pf = emb.sim.pathfinder
pos = emb.sim.get_agent(0).get_state().position
print(f"Distance to closest obstacle from agent position: {pf.distance_to_closest_obstacle(pos):.4f}m")

emb.close()
