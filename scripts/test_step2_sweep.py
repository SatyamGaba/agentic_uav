import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

sim = emb.sim
agent = sim.get_agent(0)
st = habitat_sim.AgentState()
st.position = np.array([-1.78, 3.3189, 3.82], dtype=np.float32)
half_yaw = 1.05 / 2.0
st.rotation = np.quaternion(math.cos(half_yaw), 0, math.sin(half_yaw), 0)
agent.set_state(st)
obs = emb._extract_obs()

all_obs = []
for v_i in range(4):
    d_m = obs.depth_views[v_i]
    obs_pts, _ = extract_view_obstacle_and_free_points(d_m, v_i, camera_cfg)
    if len(obs_pts) > 0:
        all_obs.append(obs_pts[:, [0, 2]])
obs_2d = np.concatenate(all_obs, axis=0) if all_obs else np.empty((0, 2))

checker = GroundCorridorChecker(robot_radius=0.10, obstacle_point_threshold=5)

# Test various forward displacements at theta = -60.0 deg
theta_rad = math.radians(-60.0)
for dist in [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00]:
    # In agent frame: forward is [-sin(theta), -cos(theta)]
    # Target point:
    px = -math.sin(theta_rad) * dist
    pz = -math.cos(theta_rad) * dist
    res = checker.evaluate(p_x=px, p_z=pz, obstacles_2d=obs_2d, m_eff=0.0)
    print(f"Target dist={dist:.2f}m (px={px:.2f}, pz={pz:.2f}): status={res.status}, cause={res.cause}, obs_count={res.collision_point_count}, rho_q={res.quantized_rho_m}")

emb.close()

