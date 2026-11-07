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
non_zero = d0[d0 > 0.0]
print("Non-zero min depth in View 0:", np.min(non_zero))
print("Histogram of non-zero depths <= 0.5m:")
for thresh in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]:
    count = np.sum((non_zero >= 0.0) & (non_zero <= thresh))
    print(f"  <= {thresh}m: {count} pixels")

pf = emb.sim.pathfinder
pos = emb.sim.get_agent(0).get_state().position
print("Agent pos:", pos)
print("Navmesh is_navigable at agent pos:", pf.is_navigable(pos))
snap = pf.snap_point(pos)
print("Snapped point:", snap, "dist:", np.linalg.norm(snap - pos))
print("Navmesh distance to closest obstacle:", pf.distance_to_closest_obstacle(pos))

# Check what object is visible in View 0 RGB
rgb0 = raw["rgb_view_0"]
print("RGB mean color:", np.mean(rgb0, axis=(0,1)))

# Let's save the RGB view to disk so we can see what it is
from PIL import Image
Image.fromarray(rgb0[:, :, :3]).save("audit/ep8628_step3_view0_rgb.png")
print("Saved RGB image to audit/ep8628_step3_view0_rgb.png")

# Now check why the navmesh let the agent get that close:
# In Habitat-Sim, sliding actuation on discrete step 'move_forward':
# When the agent steps forward against a wall, Habitat's move_filter_fn slides the agent
# along the navmesh polygon boundary until it hits the navmesh edge!
# The navmesh polygon boundary has distance_to_closest_obstacle == 0.0!
print("Agent is directly against the navmesh boundary edge.")
emb.close()
