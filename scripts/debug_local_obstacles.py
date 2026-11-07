#!/usr/bin/env python3
import sys
sys.path.insert(0, "/mnt/c/Users/Kruti/Desktop/PROG/UAV")
import json
import math
import numpy as np
from pathlib import Path
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

dataset_file = Path("agenticnav_paper_replica/data/r2rce_val_unseen_17.json")
with open(dataset_file) as f:
    ep = json.load(f)["episodes"][10]

emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], gpu_device_id=-1, max_actions=50)
obs = emb.reset(ep)

cfg = CameraConfig()
obs_list = []
for vi, dimg in obs.depth_views.items():
    pts_obs, _ = extract_view_obstacle_and_free_points(dimg, view_idx=vi, config=cfg, stride=4, y_min=0.10, y_max=1.25)
    if len(pts_obs) > 0:
        obs_list.append(pts_obs[:, [0, 2]])
obstacles_2d = np.vstack(obs_list) if obs_list else np.empty((0, 2))

checker = GroundCorridorChecker()

# Test one heading in detail: 0 deg (forward)
px = 0.0
pz = -1.5
res = checker.evaluate(p_x=px, p_z=pz, obstacles_2d=obstacles_2d)
print(f"Heading 0 deg, 1.5m: status={res.status}, cause={res.cause}, collisions={res.collision_point_count}")

# Check which obstacles are inside the corridor
theta_rad = math.atan2(px, -pz)
# Rotate obstacles into corridor frame
cos_t = math.cos(theta_rad)
sin_t = math.sin(theta_rad)
# corridor coordinates: u along path, v across path
u_obs = -obstacles_2d[:, 1] * cos_t + obstacles_2d[:, 0] * sin_t
v_obs = -obstacles_2d[:, 1] * -sin_t + obstacles_2d[:, 0] * cos_t

in_corridor = (u_obs >= 0) & (u_obs <= 1.2) & (np.abs(v_obs) <= checker.r_a)
print(f"Points inside corridor: {np.sum(in_corridor)}")
if np.sum(in_corridor) > 0:
    for u, v in zip(u_obs[in_corridor][:10], v_obs[in_corridor][:10]):
        print(f"  u={u:.2f}m, v={v:.2f}m")

# Check why: where did these obstacles come from?
# Look at height y of points from depth views
for vi, dimg in obs.depth_views.items():
    pts_obs, _ = extract_view_obstacle_and_free_points(dimg, view_idx=vi, config=cfg, stride=4, y_min=0.10, y_max=1.25)
    if len(pts_obs) > 0:
        dists = np.linalg.norm(pts_obs[:, [0, 2]], axis=1)
        close = pts_obs[dists < 1.0]
        if len(close) > 0:
            print(f"View {vi} has {len(close)} obstacles < 1.0m. Height y range: [{np.min(close[:, 1]):.3f}, {np.max(close[:, 1]):.3f}]")

emb.close()
