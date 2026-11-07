import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig, agent_to_camera_frame, reproject_point, pixel_to_normalized, 
    normalize_to_pixel, deproject_pixel, camera_to_agent_frame, convert_raw_depth,
    VIEW_AZIMUTHS_DEG, extract_view_obstacle_and_free_points
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "1868")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

agent_state = emb.sim.get_agent(0).get_state()
pos = agent_state.position
rot = agent_state.rotation
yaw = 2.0 * math.atan2(rot.y, rot.w)
goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)

path = habitat_sim.ShortestPath()
path.requested_start = pos
path.requested_end = goal_pos
emb.sim.pathfinder.find_path(path)
pts = list(path.points)

print("Pos:", pos, "yaw:", yaw)
print("Points:", pts[:4])

# Target wp 1:
target_pt = pts[1]
dx = float(target_pt[0] - pos[0])
dz = float(target_pt[2] - pos[2])
cos_y = math.cos(yaw)
sin_y = math.sin(yaw)
x_a = cos_y * dx - sin_y * dz
z_a = sin_y * dx + cos_y * dz
theta_deg = math.degrees(math.atan2(-x_a, -z_a))
print(f"dx={dx:.3f}, dz={dz:.3f}, x_a={x_a:.3f}, z_a={z_a:.3f}, heading={theta_deg:.1f}deg")

best_view = 0
min_diff = 999.0
for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
    diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
    if diff < min_diff:
        min_diff = diff
        best_view = v_idx

p_agent = np.array([x_a, 0.0, z_a], dtype=np.float64)
p_c = agent_to_camera_frame(p_agent, best_view, camera_cfg)
x0, y0, _ = reproject_point(p_c, camera_cfg)
u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
print(f"View={best_view}, raw (u, v) = ({u:.3f}, {v:.3f})")

# Extract obstacle points
checker = GroundCorridorChecker(robot_radius=0.10, obstacle_point_threshold=5)
all_obs = []
for v_i in range(4):
    depth_m = obs.depth_views[v_i]
    obs_pts, free_pts = extract_view_obstacle_and_free_points(depth_m, v_i, camera_cfg)
    if len(obs_pts) > 0:
        all_obs.append(obs_pts[:, [0, 2]])
obs_2d = np.concatenate(all_obs, axis=0) if all_obs else np.empty((0, 2))
print(f"Total obstacle points in checker: {len(obs_2d)}")

# Print depth profile along column u=0.43 (pixel x=208) in view 0
depth_img = obs.depth_views[0]
x_px = int(0.433 * 480)
print(f"Depth profile in view 0 at x={x_px}:")
for y in range(240, 480, 20):
    val = float(depth_img[y, x_px])
    print(f"  y={y} (v={y/480:.2f}): raw_val={val:.3f}")

# Check points along the shortest path
print("\nShortest path vertices projection:")
for i, pt in enumerate(pts[:8]):
    d_pt = np.array(pt) - pos
    dx = float(d_pt[0])
    dz = float(d_pt[2])
    dist = math.hypot(dx, dz)
    x_a = cos_y * dx - sin_y * dz
    z_a = sin_y * dx + cos_y * dz
    th = math.degrees(math.atan2(-x_a, -z_a))
    # view 0 check
    p_c = agent_to_camera_frame(np.array([x_a, 0.0, z_a]), 0, camera_cfg)
    if p_c[2] > 0.1:
        x0, y0, _ = reproject_point(p_c, camera_cfg)
        u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
        print(f"  wp {i}: dist={dist:.2f}m, x_a={x_a:.2f}, z_a={z_a:.2f}, heading={th:.1f}deg -> view 0 (u={u:.2f}, v={v:.2f})")
    else:
        print(f"  wp {i}: dist={dist:.2f}m behind camera in view 0")

# Check depth at (u, v)
depth_img = obs.depth_views[best_view]
x_px, y_px = normalize_to_pixel(u, v, camera_cfg)
raw_val = float(depth_img[y_px, x_px])
is_valid, z_c, err = convert_raw_depth(raw_val, camera_cfg)
print(f"x_px={x_px}, y_px={y_px}, raw_val={raw_val:.3f}, is_valid={is_valid}, z_c={z_c}, err={err}")

if is_valid:
    p_c_deproj = deproject_pixel(x_px, y_px, z_c, camera_cfg, is_normalized=False)
    p_ag_deproj = camera_to_agent_frame(p_c_deproj, best_view, camera_cfg)
    print(f"Deprojected p_ag: x={p_ag_deproj[0]:.3f}, y={p_ag_deproj[1]:.3f}, z={p_ag_deproj[2]:.3f}")
    
    # Evaluate safety
    s_res = checker.evaluate(p_x=float(p_ag_deproj[0]), p_z=float(p_ag_deproj[2]), obstacles_2d=obs_2d)
    print("Safety result:", s_res.status, s_res.cause, f"obs_count={s_res.collision_point_count}")
    print(f"Quantized: theta={s_res.quantized_theta_deg}, rho={s_res.quantized_rho_m}")

emb.close()
