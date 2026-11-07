import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points, normalize_to_pixel, deproject_pixel, camera_to_agent_frame, convert_raw_depth
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

# Move to the position after step 1: pos=[-1.78, 3.82], yaw=1.05
# Let's directly set state or step
sim = emb.sim
agent = sim.get_agent(0)
# Set agent state
st = habitat_sim.AgentState()
st.position = np.array([-1.78, 3.3189, 3.82], dtype=np.float32)
# yaw = 1.05 rad
half_yaw = 1.05 / 2.0
st.rotation = np.quaternion(math.cos(half_yaw), 0, math.sin(half_yaw), 0)
agent.set_state(st)

obs = emb._extract_obs()
print("Agent state set to:", agent.get_state().position, "rot:", agent.get_state().rotation)

# Check depth in View 1 at u=0.18, v=0.95
d1 = obs.depth_views[1]
x_px, y_px = normalize_to_pixel(0.18, 0.95, camera_cfg)
raw_val = float(d1[y_px, x_px])
is_v, z_c, err = convert_raw_depth(raw_val, camera_cfg)
print(f"View 1 at u=0.18, v=0.95: px=({x_px}, {y_px}), raw_val={raw_val:.3f}, z_c={z_c:.3f}, is_v={is_v}, err={err}")

# Extract obstacles
all_obs = []
for v_i in range(4):
    d_m = obs.depth_views[v_i]
    obs_pts, _ = extract_view_obstacle_and_free_points(d_m, v_i, camera_cfg)
    if len(obs_pts) > 0:
        all_obs.append(obs_pts[:, [0, 2]])
obs_2d = np.concatenate(all_obs, axis=0) if all_obs else np.empty((0, 2))
print("Total obstacle points:", len(obs_2d))

# Deproject
p_c = deproject_pixel(x_px, y_px, z_c, camera_cfg, is_normalized=False)
p_agent = camera_to_agent_frame(p_c, 1, camera_cfg)
print(f"p_agent: x={p_agent[0]:.3f}, y={p_agent[1]:.3f}, z={p_agent[2]:.3f}")

checker = GroundCorridorChecker(robot_radius=0.10, obstacle_point_threshold=5)
s_res = checker.evaluate(p_x=float(p_agent[0]), p_z=float(p_agent[2]), obstacles_2d=obs_2d)
print("Safety check result:", s_res.status, s_res.cause, f"obs_count={s_res.collision_point_count}")
print(f"Quantized: theta={s_res.quantized_theta_deg}, rho={s_res.quantized_rho_m}")

emb.close()
