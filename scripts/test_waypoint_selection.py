import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, numpy as np, math, habitat_sim
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    agent_to_camera_frame,
    reproject_point,
    pixel_to_normalized,
    VIEW_AZIMUTHS_DEG,
    extract_view_obstacle_and_free_points,
    normalize_to_pixel,
    deproject_pixel,
    camera_to_agent_frame,
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

dev_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"
with open(dev_path) as f:
    ep = json.load(f)["episodes"][0]
emb = HabitatGroundEmbodiment(scene_id=resolve_scene_path(ep["scene_id"]), gpu_device_id=-1)
obs = emb.reset(ep)
sim = emb.sim
pf = sim.pathfinder
agent_state = sim.get_agent(0).get_state()
pos = agent_state.position
rot = agent_state.rotation
yaw = 2.0 * math.atan2(rot.y, rot.w)

path = habitat_sim.ShortestPath()
path.requested_start = pos
path.requested_end = np.array(ep["goals"][0]["position"], dtype=np.float32)
pf.find_path(path)

camera_cfg = CameraConfig()
checker = GroundCorridorChecker(robot_radius=0.10, base_margin=0.30, obstacle_point_threshold=5)
obstacle_pts_list = []
for v_idx, d_img in obs.depth_views.items():
    pts_obs, _ = extract_view_obstacle_and_free_points(d_img, view_idx=v_idx, config=camera_cfg, stride=4, y_min=-0.73, y_max=0.37)
    if len(pts_obs) > 0:
        obstacle_pts_list.append(pts_obs)
cur_obs_agent_xyz = np.vstack(obstacle_pts_list)
obs_agent_xz = np.column_stack([cur_obs_agent_xyz[:, 0], cur_obs_agent_xyz[:, 2]])

for d_target in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
    accum = 0.0
    target_pt = path.points[-1]
    for i in range(1, len(path.points)):
        seg = float(np.linalg.norm(path.points[i] - path.points[i-1]))
        if accum + seg >= d_target:
            ratio = (d_target - accum) / seg
            target_pt = path.points[i-1] + ratio * (path.points[i] - path.points[i-1])
            break
        accum += seg
    
    dx = float(target_pt[0] - pos[0])
    dz = float(target_pt[2] - pos[2])
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x_a = cos_y * dx - sin_y * dz
    z_a = sin_y * dx + cos_y * dz
    
    theta_rad = math.atan2(-x_a, -z_a)
    theta_deg = math.degrees(theta_rad)
    best_view = 0
    min_diff = 999.0
    for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
        diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
        if diff < min_diff:
            min_diff = diff
            best_view = v_idx
            
    p_c = agent_to_camera_frame(np.array([x_a, 0.0, z_a]), best_view, camera_cfg)
    x0, y0, _ = reproject_point(p_c, camera_cfg)
    u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
    u = float(np.clip(u, 0.05, 0.95))
    v = float(np.clip(v, 0.52, 0.95))
    
    x_pix, y_pix = normalize_to_pixel(u, v, camera_cfg)
    raw_depth = float(obs.depth_views[best_view][y_pix, x_pix])
    p_c_meas = deproject_pixel(x_pix, y_pix, raw_depth, camera_cfg, is_normalized=False)
    p_a_meas = camera_to_agent_frame(p_c_meas, best_view, camera_cfg)
    
    D_raw, th_deg, rho_q, th_q_deg = checker.compute_kinematics(float(p_a_meas[0]), float(p_a_meas[2]))
    colliding, coll_count = checker.test_corridor_collision(th_q_deg, rho_q, obs_agent_xz)
    print(f"d_target={d_target:.2f}m -> view={best_view}, u={u:.2f}, v={v:.2f}, D_raw={D_raw:.2f}m, rho_q={rho_q:.2f}m, colliding={colliding}, coll_pts={coll_count}")

emb.close()

