import json
import numpy as np
import habitat_sim
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
scene_path = resolve_scene_path(ep["scene_id"])
config = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=scene_path, camera_config=config)
obs = emb.reset(ep)
obs = emb.step_discrete(2, 8)
obs = emb.step_discrete(-4, 7)

# At Step 2:
print("--- STEP 2 ---")
raw2 = emb.sim.get_sensor_observations()
for vi in [0, 1, 2, 3]:
    d = raw2[f"depth_view_{vi}"]
    nz = d[d > 0.05]
    if len(nz) > 0:
        print(f"View {vi}: min non-zero depth = {np.min(nz):.4f}m, count <= 1.0m = {np.sum(nz <= 1.0)}")
    pts, _ = extract_view_obstacle_and_free_points(d, vi, config)
    print(f"View {vi}: extracted {len(pts)} obstacle points")

# Check candidate move in Step 2:
# What if the agent commands the move (-3, 9) that led to the collision?
# In Step 2, candidate heading is -3 * 15 = -45 deg, forward distance 9 * 0.25 = 2.25m
# Let's check corridor checker on all obstacle points combined!
all_obs_pts = []
for vi in [0, 1, 2, 3]:
    pts, _ = extract_view_obstacle_and_free_points(raw2[f"depth_view_{vi}"], vi, config)
    if len(pts) > 0:
        all_obs_pts.append(pts)
combined_obs = np.vstack(all_obs_pts)[:, [0, 2]]
print(f"Total combined 2D obstacle points in Step 2: {len(combined_obs)}")

checker = GroundCorridorChecker(robot_radius=0.10, base_margin=0.30)
# Candidate target at -45 deg, 2.25m:
# p_x = 2.25 * sin(-45 deg) = 2.25 * (-0.7071) = -1.59m
# p_z = -2.25 * cos(-45 deg) = -2.25 * (0.7071) = -1.59m
res2 = checker.evaluate(p_x=-1.59, p_z=-1.59, obstacles_2d=combined_obs)
print(f"Checker evaluation of commanded move (-3, 9) at Step 2:")
print(f"  status={res2.status}, cause={res2.cause}, collision_points={res2.collision_point_count}")

emb.close()
