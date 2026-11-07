"""
Test geometric environment simulation with analytical raycasting.
Verifies that depth views produce realistic obstacle points and a crisp BEV map with corridors and walls.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
import numpy as np
from PIL import Image

from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer

def ray_segment_intersection(p, d, a, b):
    # p: ray origin (2,), d: ray direction (2,), a, b: segment endpoints (2,)
    # Ray: p + t * d, Segment: a + u * (b - a)
    v1 = p - a
    v2 = b - a
    # 2D cross product
    cross_d_v2 = d[0] * v2[1] - d[1] * v2[0]
    if abs(cross_d_v2) < 1e-8:
        return None
    t1 = (v1[0] * v2[1] - v1[1] * v2[0]) / cross_d_v2
    u1 = (v1[0] * d[1] - v1[1] * d[0]) / cross_d_v2
    if t1 > 0.05 and 0.0 <= u1 <= 1.0:
        return float(t1)
    return None

def build_corridor_walls(goal_x, goal_z):
    # Build walls around an L-shaped or straight corridor connecting (0, 0) to (goal_x, goal_z)
    walls = []
    # If goal is roughly in -Z direction:
    min_z = min(goal_z - 2.0, -12.0)
    max_z = 2.0
    min_x = min(goal_x - 2.0, -2.0)
    max_x = max(goal_x + 2.0, 2.0)
    
    # Left corridor wall
    walls.append((np.array([-1.5, max_z]), np.array([-1.5, min_z])))
    # Right corridor wall
    walls.append((np.array([1.5, max_z]), np.array([1.5, min_z])))
    # Back wall
    walls.append((np.array([-1.5, max_z]), np.array([1.5, max_z])))
    # Front end wall
    walls.append((np.array([-1.5, min_z]), np.array([1.5, min_z])))
    return walls

def generate_synthetic_obs(agent_x, agent_z, agent_yaw, walls, config):
    depth_views = {}
    rgb_views = {}
    H, W = config.image_height, config.image_width
    
    view_azimuths = {0: 0.0, 1: -90.0, 2: 180.0, 3: 90.0}
    
    for v_idx, psi_deg in view_azimuths.items():
        v_yaw = agent_yaw + math.radians(psi_deg)
        d_img = np.full((H, W), config.d_max, dtype=np.float32)
        rgb_img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Raycast each column
        for col in range(0, W, 2):
            ray_angle_local = math.atan2(col + 0.5 - config.cx, config.fx)
            total_yaw = v_yaw + ray_angle_local
            # Ray direction in world: dx = -sin(yaw), dz = -cos(yaw)
            r_dir = np.array([-math.sin(total_yaw), -math.cos(total_yaw)])
            
            # Find closest wall intersection
            min_dist = config.d_max
            p_agent = np.array([agent_x, agent_z])
            for w1, w2 in walls:
                dist = ray_segment_intersection(p_agent, r_dir, w1, w2)
                if dist is not None and dist < min_dist:
                    min_dist = dist
                    
            # For each row in this column
            for row in range(0, H, 2):
                pitch = math.atan2(row + 0.5 - config.cy, config.fy)
                if pitch > 0.02: # Points downward to floor
                    dist_floor = config.camera_height / math.tan(pitch)
                    # Optical depth z_c
                    dist_floor_opt = dist_floor * math.cos(ray_angle_local)
                    if dist_floor_opt < min_dist * math.cos(ray_angle_local):
                        # Floor hit
                        d_val = float(np.clip(dist_floor_opt, config.d_min, config.d_max))
                        d_img[row:row+2, col:col+2] = d_val
                        rgb_img[row:row+2, col:col+2] = [170, 155, 140] # Floor color
                        continue
                
                # Wall hit
                z_opt = min_dist * math.cos(ray_angle_local)
                d_val = float(np.clip(z_opt, config.d_min, config.d_max))
                d_img[row:row+2, col:col+2] = d_val
                if d_val >= config.d_max - 0.05:
                    rgb_img[row:row+2, col:col+2] = [100, 100, 110] # Far open space
                else:
                    rgb_img[row:row+2, col:col+2] = [215, 210, 205] # Wall color
                    
        depth_views[v_idx] = d_img
        rgb_views[v_idx] = rgb_img
        
    return depth_views, rgb_views

def main():
    config = CameraConfig()
    walls = build_corridor_walls(0.0, -10.0)
    
    grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
    poses = [(0.0, 0.0, 0.0), (0.0, -2.0, 0.0), (0.0, -4.0, 0.0), (0.0, -6.0, 0.0)]
    
    for px, pz, pyaw in poses:
        d_views, _ = generate_synthetic_obs(px, pz, pyaw, walls, config)
        obs_pts_list, free_pts_list = [], []
        for v, d in d_views.items():
            obs, free = extract_view_obstacle_and_free_points(d, v, config, stride=4, y_min=0.10, y_max=1.25)
            if len(obs) > 0: obs_pts_list.append(obs)
            if len(free) > 0: free_pts_list.append(free)
            
        cos_a = math.cos(pyaw)
        sin_a = math.sin(pyaw)
        if obs_pts_list:
            cur_obs = np.vstack(obs_pts_list)
            ox_w = px + cos_a * cur_obs[:, 0] + sin_a * cur_obs[:, 2]
            oz_w = pz - sin_a * cur_obs[:, 0] + cos_a * cur_obs[:, 2]
            obs_w = np.column_stack([ox_w, oz_w])
        else:
            obs_w = np.empty((0, 2))
            
        if free_pts_list:
            cur_free = np.vstack(free_pts_list)
            fx_w = px + cos_a * cur_free[:, 0] + sin_a * cur_free[:, 1]
            fz_w = pz - sin_a * cur_free[:, 0] + cos_a * cur_free[:, 1]
            free_w = np.column_stack([fx_w, fz_w])
        else:
            free_w = None
            
        grid.update(px, pz, obs_w, free_w)
        
    renderer = BEVRenderer(mode="full")
    bev_img = renderer.render(grid, node_poses=poses, current_node_idx=4)
    out_path = r"C:\Users\Kruti\.gemini\antigravity\brain\845399dd-118d-4b96-a067-6f8e6f44b148\test_corridor_bev.png"
    Image.fromarray(bev_img).save(out_path)
    print(f"Saved test corridor BEV to {out_path}")
    
    # Check pixel distribution
    unique, counts = np.unique(bev_img.reshape(-1, 3), axis=0, return_counts=True)
    for u, c in sorted(zip(unique, counts), key=lambda x: -x[1])[:10]:
        print(f"  Color {u}: {c} pixels")

if __name__ == "__main__":
    main()
