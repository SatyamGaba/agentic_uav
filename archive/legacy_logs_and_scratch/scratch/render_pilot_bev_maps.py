"""
Renders rich, high-fidelity BEV maps for the 9 pilot episodes with full environmental depth.
Extracts 3D obstacle pointclouds and 2D free-space rays from the geometric raycasting engine.
Saves them directly to the conversation artifact directory for immediate visual inspection.
"""
import os
import sys
import math
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer

ARTIFACT_DIR = r"C:\Users\Kruti\.gemini\antigravity\brain\845399dd-118d-4b96-a067-6f8e6f44b148"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Episode configurations from benchmark
EPISODES_DATA = {
    "ep01": {"goal": [-1.7, 0.0, -7.5], "poses": [(0.0, 0.0, 0.0), (0.0, -2.0, 0.0)]},
    "ep02": {"goal": [-0.9, 0.0, -9.0], "poses": [(0.0, 0.0, 0.0), (2.0, 0.0, -np.pi/2), (2.0, -2.0, 0.0), (2.0, -4.0, 0.0)]},
    "ep03": {"goal": [-0.1, 0.0, -10.5], "poses": [(0.0, 0.0, 0.0), (0.0, -2.0, 0.0), (0.0, -4.0, 0.0)]},
    "ep04": {"goal": [0.7, 0.0, -12.0], "poses": [(0.0, 0.0, 0.0), (-2.0, 0.0, np.pi/2), (-4.0, 0.0, np.pi/2)]},
    "ep05": {"goal": [1.5, 0.0, -6.0], "poses": [(0.0, 0.0, 0.0), (0.0, -2.0, 0.0), (2.0, -2.0, -np.pi/2), (4.0, -2.0, -np.pi/2)]},
    "ep06": {"goal": [2.3, 0.0, -7.5], "poses": [(0.0, 0.0, 0.0), (2.0, 0.0, -np.pi/2), (4.0, 0.0, -np.pi/2)]},
    "ep07": {"goal": [-2.5, 0.0, -9.0], "poses": [(0.0, 0.0, 0.0), (0.0, -2.0, 0.0), (2.0, -2.0, -np.pi/2), (2.0, -4.0, 0.0)]},
    "ep08": {"goal": [-1.7, 0.0, -10.5], "poses": [(0.0, 0.0, 0.0), (2.0, 0.0, -np.pi/2), (2.0, -2.0, 0.0), (2.0, -4.0, 0.0)]},
    "ep09": {"goal": [-0.9, 0.0, -12.0], "poses": [(0.0, 0.0, 0.0), (0.0, -2.0, 0.0), (-2.0, -2.0, np.pi/2)]},
}

config = ReplicaConfig()
cam_cfg = CameraConfig()
renderer = BEVRenderer(mode="full")

for ep_name, ep_info in EPISODES_DATA.items():
    goal_pos = ep_info["goal"]
    poses = ep_info["poses"]
    
    embodiment = MockGroundEmbodiment(camera_config=cam_cfg)
    embodiment.reset({"goal_position": goal_pos})
    
    grid = OccupancyGrid2D(grid_size=config.bev_grid_size, resolution_m=config.bev_resolution_m)
    
    for px, pz, pyaw in poses:
        embodiment.x = px
        embodiment.z = pz
        embodiment.yaw_rad = pyaw
        obs = embodiment._make_obs()
        
        obstacle_pts_list = []
        free_pts_list = []
        for v_idx, d_img in obs.depth_views.items():
            pts_obs, pts_free = extract_view_obstacle_and_free_points(
                d_img,
                view_idx=v_idx,
                config=cam_cfg,
                stride=4,
                y_min=config.chassis_y_min,
                y_max=config.chassis_y_max,
            )
            if len(pts_obs) > 0:
                obstacle_pts_list.append(pts_obs)
            if len(pts_free) > 0:
                free_pts_list.append(pts_free)
                
        cos_a = math.cos(pyaw)
        sin_a = math.sin(pyaw)
        
        if obstacle_pts_list:
            cur_obs_agent_xyz = np.vstack(obstacle_pts_list)
            ox_agent = cur_obs_agent_xyz[:, 0]
            oz_agent = cur_obs_agent_xyz[:, 2]
            ox_world = px + cos_a * ox_agent + sin_a * oz_agent
            oz_world = pz - sin_a * ox_agent + cos_a * oz_agent
            obs_world_xz = np.column_stack([ox_world, oz_world])
        else:
            obs_world_xz = np.empty((0, 2), dtype=np.float64)
            
        if free_pts_list:
            cur_free_agent_xz = np.vstack(free_pts_list)
            fx_agent = cur_free_agent_xz[:, 0]
            fz_agent = cur_free_agent_xz[:, 1]
            fx_world = px + cos_a * fx_agent + sin_a * fz_agent
            fz_world = pz - sin_a * fx_agent + cos_a * fz_agent
            free_world_xz = np.column_stack([fx_world, fz_world])
        else:
            free_world_xz = None
            
        grid.update(
            agent_x_m=px,
            agent_z_m=pz,
            obstacle_points_world_xz=obs_world_xz,
            free_endpoints_world_xz=free_world_xz,
        )
        
    bev_img = renderer.render(grid, node_poses=poses, current_node_idx=len(poses))
    out_file = os.path.join(ARTIFACT_DIR, f"bev_map_{ep_name}.png")
    # Upscale 2x with NEAREST for crisp display
    pil_img = Image.fromarray(bev_img).resize((512, 512), resample=Image.Resampling.NEAREST)
    pil_img.save(out_file)
    print(f"Saved rich BEV map: {out_file}")

print("All 9 BEV maps successfully re-rendered with full environmental depth!")
