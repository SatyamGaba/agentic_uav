import json
import math
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import (
    HabitatGroundEmbodiment,
    resolve_scene_path,
)
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    unproject_depth_image,
    extract_view_obstacle_and_free_points,
    deproject_pixel,
    camera_to_agent_frame,
    normalize_to_pixel,
    convert_raw_depth,
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

def diagnose_step():
    dev_path = PROJECT_ROOT / "agenticnav_paper_replica/data/r2rce_dev_20.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ep = data.get("episodes", data)[0]
    
    config = ReplicaConfig()
    res_scene = resolve_scene_path(str(ep["scene_id"]))
    emb = HabitatGroundEmbodiment(scene_id=res_scene, gpu_device_id=-1)
    obs = emb.reset(ep)
    
    camera_cfg = CameraConfig()
    checker = GroundCorridorChecker(
        robot_radius=0.10,
        base_margin=0.30,
        obstacle_point_threshold=5,
    )
    
    # Extract obstacle point cloud as runner does
    obstacle_pts_list = []
    for v_idx, d_img in obs.depth_views.items():
        pts_obs, _ = extract_view_obstacle_and_free_points(
            d_img,
            view_idx=v_idx,
            config=camera_cfg,
            stride=4,
            y_min=config.chassis_y_min,
            y_max=config.chassis_y_max,
        )
        if len(pts_obs) > 0:
            obstacle_pts_list.append(pts_obs)
            
    cur_obs_agent_xyz = np.vstack(obstacle_pts_list)
    obs_agent_xz = np.column_stack([cur_obs_agent_xyz[:, 0], cur_obs_agent_xyz[:, 2]])
    
    print(f"Total obstacle points in agent frame: {len(cur_obs_agent_xyz)}")
    print(f"Y min in cloud: {np.min(cur_obs_agent_xyz[:, 1]):.3f}m, Y max: {np.max(cur_obs_agent_xyz[:, 1]):.3f}m")
    
    # Target from scripted oracle: view=0, u=0.86, v=0.95
    u, v, view = 0.86, 0.95, 0
    depth_img = obs.depth_views[view]
    x0, y0 = normalize_to_pixel(u, v, camera_cfg)
    raw_val = float(depth_img[y0, x0])
    print(f"Target pixel: ({x0}, {y0}), raw_depth: {raw_val:.3f}m")
    
    p_c = deproject_pixel(x0, y0, raw_val, camera_cfg, is_normalized=False)
    p_agent = camera_to_agent_frame(p_c, view, camera_cfg)
    print(f"p_agent: {p_agent}")
    
    # Evaluate kinematics
    D_raw, theta_deg, rho_q, theta_q_deg = checker.compute_kinematics(float(p_agent[0]), float(p_agent[2]))
    print(f"D_raw: {D_raw:.2f}m, theta_deg: {theta_deg:.1f}deg, rho_q: {rho_q:.2f}m, theta_q_deg: {theta_q_deg:.1f}deg")
    
    # Test collision
    colliding, coll_count = checker.test_corridor_collision(theta_q_deg, rho_q, obs_agent_xz)
    print(f"Corridor check: colliding={colliding}, collision_points={coll_count} (threshold={checker.obstacle_point_threshold})")
    
    # If colliding, print the colliding points
    theta_rad = math.radians(theta_q_deg)
    sin_th = math.sin(theta_rad)
    cos_th = math.cos(theta_rad)
    ox = obs_agent_xz[:, 0]
    oz = obs_agent_xz[:, 1]
    s = -ox * sin_th - oz * cos_th
    h = ox * cos_th - oz * sin_th
    in_s = (s > 0.0) & (s < (rho_q + checker.r_a))
    in_h = np.abs(h) < checker.r_a
    mask = in_s & in_h
    
    colliding_xyz = cur_obs_agent_xyz[mask]
    print(f"\nColliding obstacle points (first 10):")
    for pt in colliding_xyz[:10]:
        print(f"  x={pt[0]:.3f}, y={pt[1]:.3f}, z={pt[2]:.3f}")
        
    emb.close()

if __name__ == "__main__":
    diagnose_step()

