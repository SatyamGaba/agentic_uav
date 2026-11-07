import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, agent_to_camera_frame, reproject_point, pixel_to_normalized, VIEW_AZIMUTHS_DEG
from agenticnav_paper_replica.harness.runner import run_episode

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

config = ReplicaConfig()
config.max_simulator_actions = 500
config.max_decision_steps = 25
config.robot_radius = 0.10
config.obstacle_point_threshold = 5

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)

def make_oracle_client():
    reselect_attempts = [0]
    step_num = [0]
    def oracle_client(messages, tools):
        step_num[0] += 1
        last_msg = messages[-1].get("content", "") if messages else ""
        is_reselect = isinstance(last_msg, str) and any(w in last_msg.lower() for w in ["reselect", "collision", "rejected", "blocked"])
        if is_reselect:
            reselect_attempts[0] += 1
        else:
            reselect_attempts[0] = 0

        sim = emb.sim
        pf = sim.pathfinder
        agent_state = sim.get_agent(0).get_state()
        pos = agent_state.position
        rot = agent_state.rotation
        yaw = 2.0 * math.atan2(rot.y, rot.w)
        goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)

        path = habitat_sim.ShortestPath()
        path.requested_start = pos
        path.requested_end = goal_pos
        found = pf.find_path(path)
        dist = path.geodesic_distance if found else float(np.linalg.norm(goal_pos - pos))

        if dist <= 2.5:
            return {"thought": "stop", "tool_calls": [{"name": "stop", "arguments": {"reason": "Reached"}}]}

        pts = list(path.points)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)

        # Look along path for best visible waypoint
        # If reselection, try slightly different distance
        target_d_list = [1.8, 1.5, 2.0, 1.2, 0.9]
        t_idx = min(reselect_attempts[0], len(target_d_list) - 1)
        target_d = min(target_d_list[t_idx], dist)

        accum = 0.0
        target_pt = pts[-1]
        for i in range(1, len(pts)):
            seg = float(np.linalg.norm(pts[i] - pts[i-1]))
            if accum + seg >= target_d:
                ratio = (target_d - accum) / max(seg, 1e-6)
                target_pt = pts[i-1] + ratio * (pts[i] - pts[i-1])
                break
            accum += seg

        dx = float(target_pt[0] - pos[0])
        dz = float(target_pt[2] - pos[2])
        x_a = cos_y * dx - sin_y * dz
        z_a = sin_y * dx + cos_y * dz
        theta_rad = math.atan2(-x_a, -z_a)
        theta_deg = math.degrees(theta_rad)

        best_view = 0
        min_diff = 999.0
        for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
            diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
            if diff < min_diff:
                min_diff = diff
                best_view = v_idx

        # In best_view frame:
        # Pinhole projection for a floor point:
        # y_v = 0.0 -> y_c = camera_height = 1.25m
        # Let's project target_pt into camera optical frame
        p_agent = np.array([x_a, 0.0, z_a], dtype=np.float64)
        p_c = agent_to_camera_frame(p_agent, best_view, camera_cfg)
        x_c, y_c, z_c = p_c[0], p_c[1], p_c[2]

        # Horizontal coordinate
        u = 0.5 + (camera_cfg.fx / camera_cfg.image_width) * (x_c / max(z_c, 0.1))
        # If z_c is close, pick v in the lower half of the image
        # In Habitat, floor at distance z_c has row = cy + fy * 1.25 / z_c
        # If that row > 470, use row = 460 (v = 0.95), but query depth will be looked up
        v = 0.5 + (camera_cfg.fy / camera_cfg.image_height) * (y_c / max(z_c, 0.1))

        u = float(np.clip(u, 0.05, 0.95))
        v = float(np.clip(v, 0.52, 0.95))

        print(f"  [Step {step_num[0]}] dist={dist:.2f}m, target_d={target_d:.2f}m, theta={theta_deg:.1f}deg -> view={best_view}, u={u:.2f}, v={v:.2f}", flush=True)
        return {
            "thought": "move",
            "tool_calls": [{"name": "move_to", "arguments": {"view": best_view, "u": round(u, 2), "v": round(v, 2)}}]
        }
    return oracle_client

res = run_episode(episode_data=ep, embodiment=emb, config=config, vlm_client_fn=make_oracle_client())
print("FINISHED EP 8628:")
print(f"Success={res.metrics.success}, SPL={res.metrics.spl:.4f}, NE={res.metrics.navigation_error_m:.2f}m, Actions={res.metrics.simulator_action_count}, Steps={res.total_steps}, Cat={res.failure_category}")
emb.close()

