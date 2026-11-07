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
    def oracle_client(messages, tools):
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

        # Search along path for a waypoint whose projection onto one of the 4 views is fully valid and on-screen
        # Target distance candidates: from 2.0m down to 1.7m (on-screen floor), or if shorter, turn-only or visible point
        found_target = None
        target_candidates = [2.0, 1.8, 1.5, 2.2, 1.2, 2.5]
        # Shift candidate based on reselect
        if reselect_attempts[0] > 0:
            target_candidates = target_candidates[reselect_attempts[0] % len(target_candidates):] + target_candidates[:reselect_attempts[0] % len(target_candidates)]

        for target_d in target_candidates:
            target_d = min(target_d, dist)
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
            p_agent = np.array([x_a, 0.0, z_a], dtype=np.float64)
            theta_deg = math.degrees(math.atan2(-x_a, -z_a))

            # Find best view
            best_v = 0
            min_d = 999.0
            for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
                diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
                if diff < min_d:
                    min_d = diff
                    best_v = v_idx

            p_c = agent_to_camera_frame(p_agent, best_v, camera_cfg)
            if p_c[2] > 0.5:
                x0, y0, _ = reproject_point(p_c, camera_cfg)
                u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
                # Check if naturally in bounds (0.05 <= u <= 0.95 and 0.50 <= v <= 0.95)
                if 0.05 <= u <= 0.95 and 0.50 <= v <= 0.95:
                    found_target = (best_v, u, v, target_d, theta_deg)
                    break

        if found_target is not None:
            best_view, u, v, td, th = found_target
        else:
            # Fallback: take standard target 1.8m
            td = min(1.8, dist)
            # Find point at 1.8m
            accum = 0.0
            target_pt = pts[-1]
            for i in range(1, len(pts)):
                seg = float(np.linalg.norm(pts[i] - pts[i-1]))
                if accum + seg >= td:
                    ratio = (td - accum) / max(seg, 1e-6)
                    target_pt = pts[i-1] + ratio * (pts[i] - pts[i-1])
                    break
                accum += seg
            dx = float(target_pt[0] - pos[0])
            dz = float(target_pt[2] - pos[2])
            x_a = cos_y * dx - sin_y * dz
            z_a = sin_y * dx + cos_y * dz
            p_agent = np.array([x_a, 0.0, z_a], dtype=np.float64)
            theta_deg = math.degrees(math.atan2(-x_a, -z_a))
            best_view = 0
            min_d = 999.0
            for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
                diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
                if diff < min_d:
                    min_d = diff
                    best_view = v_idx
            p_c = agent_to_camera_frame(p_agent, best_view, camera_cfg)
            x0, y0, _ = reproject_point(p_c, camera_cfg)
            u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
            u = float(np.clip(u, 0.08, 0.92))
            v = float(np.clip(v, 0.55, 0.92))

        return {
            "thought": "move",
            "tool_calls": [{"name": "move_to", "arguments": {"view": best_view, "u": round(u, 2), "v": round(v, 2)}}]
        }
    return oracle_client

res = run_episode(episode_data=ep, embodiment=emb, config=config, vlm_client_fn=make_oracle_client())
print("FINISHED EP 8628:")
print(f"Success={res.metrics.success}, SPL={res.metrics.spl:.4f}, NE={res.metrics.navigation_error_m:.2f}m, Actions={res.metrics.simulator_action_count}, Steps={res.total_steps}, Cat={res.failure_category}")
emb.close()

