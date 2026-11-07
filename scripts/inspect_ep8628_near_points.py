import json
import math
import numpy as np
import habitat_sim

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)

ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
scene_path = resolve_scene_path(ep["scene_id"])
cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=scene_path, camera_config=cfg)

obs = emb.reset(ep)
print("Start position:", obs.position)
print("Start pose:", obs.allocentric_pose)

# In view 0, look at the depth image
depth_v0 = obs.depth_views[0]
print("Depth view 0 min:", np.min(depth_v0), "max:", np.max(depth_v0))
print("Pixels exactly <= 0.501m:", np.sum(depth_v0 <= 0.501))
print("Pixels between 0.501 and 0.60m:", np.sum((depth_v0 > 0.501) & (depth_v0 <= 0.60)))

# Unproject ALL points without masking
h, w = depth_v0.shape
y_idx, x_idx = np.indices((h, w))
z_c = depth_v0
x_c = (x_idx - cfg.cx) * z_c / cfg.fx
y_c = (y_idx - cfg.cy) * z_c / cfg.fy

# In agent frame (view 0 is front, yaw=0):
X_ag = x_c
Y_ag = -y_c + cfg.camera_height
Z_ag = -z_c

near_mask = (depth_v0 <= 0.501)
print(f"Near points count: {np.sum(near_mask)}")
if np.sum(near_mask) > 0:
    print(f"Near points Y_agent min: {np.min(Y_ag[near_mask]):.3f}m, max: {np.max(Y_ag[near_mask]):.3f}m")
    print(f"Near points X_agent min: {np.min(X_ag[near_mask]):.3f}m, max: {np.max(X_ag[near_mask]):.3f}m")
    print(f"Near points Z_agent min: {np.min(Z_ag[near_mask]):.3f}m, max: {np.max(Z_ag[near_mask]):.3f}m")
    
    sim = emb.sim
    pf = sim.pathfinder
    
    sample_indices = np.where(near_mask)
    ax, az, ayaw = obs.allocentric_pose
    print(f"Agent allocentric pose: x={ax:.3f}, z={az:.3f}, yaw={ayaw:.3f}")
    for i in range(min(5, len(sample_indices[0]))):
        r = sample_indices[0][i]
        c = sample_indices[1][i]
        x_a = X_ag[r, c]
        y_a = Y_ag[r, c]
        z_a = Z_ag[r, c]
        # In Habitat: world position = agent_pos + rot * pt
        # Agent forward is -Z_ag, Right is +X_ag
        cos_y = math.cos(ayaw)
        sin_y = math.sin(ayaw)
        # Allocentric: dx = x_a * cos - z_a * sin?
        # Let's check simulator state
        agent_state = sim.get_agent(0).get_state()
        pos = agent_state.position
        print(f"  Pixel (r={r}, c={c}): depth={depth_v0[r, c]:.4f}m, agent_xyz=({x_a:.3f}, {y_a:.3f}, {z_a:.3f})")

# Also check views 1, 2, 3
for vi in [1, 2, 3]:
    d_v = obs.depth_views[vi]
    print(f"View {vi} min={np.min(d_v):.3f}m, <=0.501m count={np.sum(d_v <= 0.501)}")

emb.close()
