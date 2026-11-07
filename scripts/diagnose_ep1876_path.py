import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, numpy as np, math, habitat_sim
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, normalize_to_pixel, deproject_pixel, camera_to_agent_frame, VIEW_AZIMUTHS_DEG, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

dev_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"
with open(dev_path) as f:
    ep = json.load(f)["episodes"][3]

emb = HabitatGroundEmbodiment(scene_id=resolve_scene_path(ep["scene_id"]), gpu_device_id=-1)
obs = emb.reset(ep)
obs = emb.step_discrete(turn_steps_15deg=0, forward_steps_025m=7)

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
print("Path points:", len(path.points))
for i, pt in enumerate(path.points):
    print(f"pt {i}: {pt}")

camera_cfg = CameraConfig()
checker = GroundCorridorChecker(robot_radius=0.10, base_margin=0.30, obstacle_point_threshold=5)
obstacle_pts_list = []
for v_idx, d_img in obs.depth_views.items():
    pts_obs, _ = extract_view_obstacle_and_free_points(d_img, view_idx=v_idx, config=camera_cfg, stride=4, y_min=0.15, y_max=1.25)
    if len(pts_obs) > 0: obstacle_pts_list.append(pts_obs)
cur_obs_agent_xyz = np.vstack(obstacle_pts_list)
obs_agent_xz = np.column_stack([cur_obs_agent_xyz[:, 0], cur_obs_agent_xyz[:, 2]])

# Check why u=0.60, v=0.95 or other points collide
for u_cand in [0.50, 0.55, 0.60, 0.65, 0.70]:
    for v_cand in [0.95, 0.90, 0.85, 0.80]:
        x0, y0 = normalize_to_pixel(u_cand, v_cand, camera_cfg)
        raw_val = float(obs.depth_views[0][y0, x0])
        p_c = deproject_pixel(x0, y0, raw_val, camera_cfg, is_normalized=False)
        p_a = camera_to_agent_frame(p_c, 0, camera_cfg)
        D_raw, th, rho, th_q = checker.compute_kinematics(float(p_a[0]), float(p_a[2]))
        coll, cnt = checker.test_corridor_collision(th_q, rho, obs_agent_xz)
        print(f"u={u_cand:.2f}, v={v_cand:.2f}: rho={rho:.2f}m, th_q={th_q:.1f}deg, coll={coll}, cnt={cnt}")

emb.close()

