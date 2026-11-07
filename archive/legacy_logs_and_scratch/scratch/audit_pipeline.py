import sys
import os
import math
import json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer
from agenticnav_paper_replica.harness.prompts import SYSTEM_PROMPT, build_step_user_prompt

print("=== DEEP PIPELINE AUDIT FOR LOCAL TEST CASE ===")

config = ReplicaConfig()
ep_data = {
    "episode_id": "audit_ep01",
    "instruction": "Walk forward down corridor and stop.",
    "start_position": [0.0, 0.0, 0.0],
    "goal_position": [0.0, 0.0, -10.0],
    "shortest_path_distance": 10.0
}
emb = MockGroundEmbodiment(max_actions=100)
obs = emb.reset(ep_data)

print(f"1. Embodiment: {type(emb).__name__}")
print(f"   Initial Agent Pose: x={emb.x}, z={emb.z}, yaw={math.degrees(emb.yaw_rad):.1f}°")
print(f"   Walls defined in embodiment: {len(emb.walls)} segments")
for i, (w1, w2) in enumerate(emb.walls):
    print(f"     Wall {i}: [{w1[0]:.1f}, {w1[1]:.1f}] -> [{w2[0]:.1f}, {w2[1]:.1f}]")

print(f"\n2. Sensor Views (Depth & RGB):")
for v, d_img in obs.depth_views.items():
    rgb_img = obs.rgb_views[v]
    unique_c = len(np.unique(rgb_img.reshape(-1, 3), axis=0))
    print(f"   View {v}: Depth [min={d_img.min():.2f}m, max={d_img.max():.2f}m] | RGB unique colors = {unique_c}")

# Unproject
camera_cfg = CameraConfig()
occ_grid = OccupancyGrid2D(grid_size=config.bev_grid_size, resolution_m=config.bev_resolution_m)
bev_renderer = BEVRenderer(mode="full")

obstacle_pts_list = []
free_pts_list = []
for v_idx, d_img in obs.depth_views.items():
    pts_obs, pts_free = extract_view_obstacle_and_free_points(
        d_img, view_idx=v_idx, config=camera_cfg, stride=4,
        y_min=config.chassis_y_min, y_max=config.chassis_y_max
    )
    if len(pts_obs) > 0:
        obstacle_pts_list.append(pts_obs)
    if len(pts_free) > 0:
        free_pts_list.append(pts_free)

ax, az, ayaw = obs.allocentric_pose
cos_a = math.cos(ayaw)
sin_a = math.sin(ayaw)

if obstacle_pts_list:
    cur_obs_agent_xyz = np.vstack(obstacle_pts_list)
    ox_agent = cur_obs_agent_xyz[:, 0]
    oz_agent = cur_obs_agent_xyz[:, 2]
    ox_world = ax + cos_a * ox_agent + sin_a * oz_agent
    oz_world = az - sin_a * ox_agent + cos_a * oz_agent
    obs_world_xz = np.column_stack([ox_world, oz_world])
else:
    obs_world_xz = np.empty((0, 2), dtype=np.float64)

if free_pts_list:
    cur_free_agent_xz = np.vstack(free_pts_list)
    fx_agent = cur_free_agent_xz[:, 0]
    fz_agent = cur_free_agent_xz[:, 1]
    fx_world = ax + cos_a * fx_agent + sin_a * fz_agent
    fz_world = az - sin_a * fx_agent + cos_a * fz_agent
    free_world_xz = np.column_stack([fx_world, fz_world])
else:
    free_world_xz = None

occ_grid.update(
    agent_x_m=ax,
    agent_z_m=az,
    obstacle_points_world_xz=obs_world_xz,
    free_endpoints_world_xz=free_world_xz,
)

occ_img = occ_grid.to_uint8_image()
free_count = np.sum(occ_img == 255)
occ_count = np.sum(occ_img == 0)
unk_count = np.sum(occ_img == 127)
print(f"\n3. Occupancy Grid:")
print(f"   Grid Size: {occ_grid.grid_size}x{occ_grid.grid_size} @ {occ_grid.resolution}m/px ({occ_grid.grid_size * occ_grid.resolution:.1f}m span)")
print(f"   Free Cells: {free_count} ({free_count * (occ_grid.resolution**2):.2f} m²), Obstacle Cells: {occ_count}, Unexplored: {unk_count}")

# Render BEV
bev_img = bev_renderer.render(occ_grid, node_poses=[(ax, az, ayaw)], current_node_idx=1)
print(f"\n4. BEV Renderer:")
print(f"   Rendered shape: {bev_img.shape}, dtype={bev_img.dtype}")
print(f"   Unique pixel values: {len(np.unique(bev_img.reshape(-1, 3), axis=0))}")
Image.fromarray(bev_img).save("scratch/audit_bev_test.png")
print("   Saved scratch/audit_bev_test.png")

# Now let's test a step forward: move forward 2.5m
obs_next = emb.step_discrete(turn_steps_15deg=0, forward_steps_025m=10)
print(f"\n5. Step Discrete (Odometry):")
print(f"   Moved: {obs_next.actual_distance_moved:.2f}m")
print(f"   New Pose: x={emb.x:.2f}, z={emb.z:.2f}, yaw={math.degrees(emb.yaw_rad):.1f}°")
print(f"   Collided: {obs_next.collided}")
print(f"   Distance to goal: {emb.get_geodesic_distance_to_goal():.2f}m")
