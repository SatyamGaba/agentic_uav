import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

sim = emb.sim
agent = sim.get_agent(0)
st = habitat_sim.AgentState()
st.position = np.array([-1.78, 3.3189, 3.82], dtype=np.float32)
half_yaw = 1.05 / 2.0
st.rotation = np.quaternion(math.cos(half_yaw), 0, math.sin(half_yaw), 0)
agent.set_state(st)
obs = emb._extract_obs()

all_obs = []
for v_i in range(4):
    d_m = obs.depth_views[v_i]
    obs_pts, _ = extract_view_obstacle_and_free_points(d_m, v_i, camera_cfg)
    if len(obs_pts) > 0:
        all_obs.append(obs_pts[:, [0, 2]])
obs_2d = np.concatenate(all_obs, axis=0) if all_obs else np.empty((0, 2))

theta_rad = math.radians(-60.0)
# Look at corridor for dist=0.75
b_th = np.array([-math.sin(theta_rad), -math.cos(theta_rad)])
l_th = np.array([math.cos(theta_rad), -math.sin(theta_rad)])

# Project obstacle points onto b_th (s) and l_th (h)
s_coords = obs_2d @ b_th
h_coords = obs_2d @ l_th

in_corridor = (s_coords > 0.0) & (s_coords < 0.75 + 0.10) & (np.abs(h_coords) < 0.10)
pts_in = obs_2d[in_corridor]
print(f"Points in corridor for rho=0.75m: {len(pts_in)}")
print("Sample s coordinates of colliding points:")
print(np.sort(s_coords[in_corridor])[:10])
print("Sample (x, z) coordinates of colliding points in agent frame:")
print(pts_in[:5])

# Convert to world frame:
# pos = [-1.78, 3.32, 3.82], yaw = 1.05
# x_w = pos[0] + cos_y * x_a + sin_y * z_a
# z_w = pos[2] - sin_y * x_a + cos_y * z_a
cos_y = math.cos(1.05)
sin_y = math.sin(1.05)
x_w = -1.78 + cos_y * pts_in[:, 0] + sin_y * pts_in[:, 1]
z_w = 3.82 - sin_y * pts_in[:, 0] + cos_y * pts_in[:, 1]
print("World coordinates of collision points:")
for i in range(min(5, len(x_w))):
    print(f"  x={x_w[i]:.3f}, z={z_w[i]:.3f}")

# Check navmesh clearance at these points
pf = sim.pathfinder
for i in range(min(5, len(x_w))):
    pt_3d = np.array([x_w[i], 3.3189, z_w[i]], dtype=np.float32)
    is_nav = pf.is_navigable(pt_3d)
    snap = pf.snap_point(pt_3d)
    print(f"  navigable={is_nav}, dist to navmesh={np.linalg.norm(snap - pt_3d):.3f}")

emb.close()

