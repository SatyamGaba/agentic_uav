import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points, deproject_pixel, camera_to_agent_frame, convert_raw_depth
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
camera_cfg = CameraConfig()
config = ReplicaConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
# Step 1
obs = emb.step_discrete(2, 8)

# Extract obstacle points
obstacle_pts_list = []
for v_idx, d_img in obs.depth_views.items():
    pts_obs, pts_free = extract_view_obstacle_and_free_points(
        d_img,
        view_idx=v_idx,
        config=camera_cfg,
        stride=4,
        y_min=config.chassis_y_min,
        y_max=config.chassis_y_max,
    )
    if len(pts_obs) > 0:
        obstacle_pts_list.append(pts_obs)

obs_agent_xyz = np.vstack(obstacle_pts_list) if obstacle_pts_list else np.empty((0, 3))
obs_agent_xz = obs_agent_xyz[:, [0, 2]]

checker = GroundCorridorChecker(robot_radius=0.10, base_margin=0.30, rho_max=2.5, obstacle_point_threshold=5)

x0, y0 = int(round(0.18 * 640)), int(round(0.95 * 480))
raw_val = float(obs.depth_views[1][y0, x0])
_, z_c, _ = convert_raw_depth(raw_val, camera_cfg)
p_c = deproject_pixel(x0, y0, z_c, camera_cfg, is_normalized=False)
p_agent = camera_to_agent_frame(p_c, 1, camera_cfg)
p_x, p_z = float(p_agent[0]), float(p_agent[2])

D_raw, theta_deg, rho_q, theta_q_deg = checker.compute_kinematics(p_x, p_z, m_eff=0.30)
print(f"p_x={p_x:.2f}, p_z={p_z:.2f}, D_raw={D_raw:.2f}, theta_deg={theta_deg:.1f}, theta_q_deg={theta_q_deg:.1f}, rho_q={rho_q:.2f}")

# Corridor check
theta_rad = math.radians(theta_q_deg)
sin_th = math.sin(theta_rad)
cos_th = math.cos(theta_rad)
ox = obs_agent_xz[:, 0]
oz = obs_agent_xz[:, 1]
s = -ox * sin_th - oz * cos_th
h = ox * cos_th - oz * sin_th
in_s = (s > 0.0) & (s < (rho_q + 0.10))
in_h = np.abs(h) < 0.10
in_corr = in_s & in_h
coll_pts_xyz = obs_agent_xyz[in_corr]
print(f"Number of points in corridor: {len(coll_pts_xyz)}")
for pt in coll_pts_xyz[:15]:
    print(f"  pt: x={pt[0]:.3f}, y={pt[1]:.3f}, z={pt[2]:.3f}")

emb.close()
