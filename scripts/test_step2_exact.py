import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

# Step 1: executed move
obs = emb.step_discrete(turn_steps_15deg=2, forward_steps_025m=8)
print("After Step 1: pos=", emb.sim.get_agent(0).get_state().position, "rot=", emb.sim.get_agent(0).get_state().rotation)

# Now check Step 2:
all_obs = []
for v_i in range(4):
    d_m = obs.depth_views[v_i]
    obs_pts, _ = extract_view_obstacle_and_free_points(d_m, v_i, camera_cfg)
    if len(obs_pts) > 0:
        all_obs.append(obs_pts[:, [0, 2]])
obs_2d = np.concatenate(all_obs, axis=0) if all_obs else np.empty((0, 2))
print("Total obstacle points:", len(obs_2d))

cache = EpisodicImageCache()
checker = GroundCorridorChecker(robot_radius=0.10, obstacle_point_threshold=5)
executor = ToolExecutor(camera_cfg, checker, cache)

res = executor.execute_move_to(
    view=1, u=0.18, v=0.95, reason="Step 2 test",
    depth_views=obs.depth_views,
    obstacle_points_agent_xz=obs_2d
)
print("Move result:", res)

# If rejected, print the colliding points:
# What was the quantized motion?
# Check raw kinematics:
d_img = obs.depth_views[1]
from agenticnav_paper_replica.geometry.pinhole import normalize_to_pixel, convert_raw_depth, deproject_pixel, camera_to_agent_frame
x0, y0 = normalize_to_pixel(0.18, 0.95, camera_cfg)
raw_v = float(d_img[y0, x0])
_, z_c, _ = convert_raw_depth(raw_v, camera_cfg)
p_c = deproject_pixel(x0, y0, z_c, camera_cfg, is_normalized=False)
p_ag = camera_to_agent_frame(p_c, 1, camera_cfg)
p_x, p_z = float(p_ag[0]), float(p_ag[2])
D_raw, theta_deg, rho_q, theta_q_deg = checker.compute_kinematics(p_x, p_z, checker.base_margin)
print(f"p_x={p_x:.3f}, p_z={p_z:.3f}, D_raw={D_raw:.2f}, theta={theta_deg:.1f}, rho_q={rho_q:.2f}, theta_q={theta_q_deg:.1f}")

# Check points in corridor
th_rad = math.radians(theta_q_deg)
sin_th = math.sin(th_rad)
cos_th = math.cos(th_rad)
ox = obs_2d[:, 0]
oz = obs_2d[:, 1]
s = -ox * sin_th - oz * cos_th
h = ox * cos_th - oz * sin_th
in_s = (s > 0.0) & (s < (rho_q + checker.r_a))
in_h = np.abs(h) < checker.r_a
coll = in_s & in_h
print(f"Colliding count: {np.sum(coll)}")
if np.sum(coll) > 0:
    coll_pts = obs_2d[coll]
    print("Sample colliding points in agent frame [x, z]:")
    print(coll_pts[:10])
    print("Distances s of colliding points:")
    print(np.sort(s[coll])[:10])

emb.close()

