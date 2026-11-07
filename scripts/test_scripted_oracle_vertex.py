import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, agent_to_camera_frame, reproject_point, pixel_to_normalized, VIEW_AZIMUTHS_DEG
from agenticnav_paper_replica.harness.runner import run_episode

def test_ep(ep_id_str):
    with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
        dev = json.load(f)
    ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == ep_id_str)

    config = ReplicaConfig()
    config.max_simulator_actions = 500
    config.max_decision_steps = 25
    config.robot_radius = 0.10
    config.obstacle_point_threshold = 5

    camera_cfg = CameraConfig()
    emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)

    def make_oracle_client():
        step_num = [0]
        reselect_count = [0]
        def oracle_client(messages, tools):
            step_num[0] += 1
            last_msg = messages[-1].get("content", "") if messages else ""
            is_reselect = isinstance(last_msg, str) and any(w in last_msg.lower() for w in ["reselect", "collision", "rejected", "blocked"])
            if is_reselect:
                reselect_count[0] += 1
            else:
                reselect_count[0] = 0

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

            # Follow PathFinder's collision-free polygonal chain:
            # Pick a target point along the FIRST segment (or subsequent segments if strictly line-of-sight clear)
            # Find the furthest point along the path that has direct line of sight from pos
            target_pt = pts[1]
            # If the segment from pos to pts[1] is short (< 0.4m) and there are more points, advance to next
            seg_len = float(np.linalg.norm(pts[1] - pos))
            if seg_len < 0.35 and len(pts) > 2:
                target_pt = pts[2]
            elif seg_len > 2.0:
                # Clamp segment target to 2.0m max step
                ratio = 2.0 / seg_len
                target_pt = pos + ratio * (pts[1] - pos)

            # If reselecting, nudge target slightly back along segment
            if reselect_count[0] > 0:
                pull_back = 0.20 * reselect_count[0]
                vec = target_pt - pos
                d_vec = np.linalg.norm(vec)
                if d_vec > pull_back + 0.25:
                    target_pt = pos + (d_vec - pull_back) * (vec / d_vec)

            dx = float(target_pt[0] - pos[0])
            dz = float(target_pt[2] - pos[2])
            x_a = cos_y * dx - sin_y * dz
            z_a = sin_y * dx + cos_y * dz
            theta_deg = math.degrees(math.atan2(-x_a, -z_a))

            best_view = 0
            min_diff = 999.0
            for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
                diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
                if diff < min_diff:
                    min_diff = diff
                    best_view = v_idx

            # Reproject to camera coordinates
            p_agent = np.array([x_a, 0.0, z_a], dtype=np.float64)
            p_c = agent_to_camera_frame(p_agent, best_view, camera_cfg)
            x0, y0, _ = reproject_point(p_c, camera_cfg)
            u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
            u = float(np.clip(u, 0.05, 0.95))
            v = float(np.clip(v, 0.52, 0.95))

            return {
                "thought": "move",
                "tool_calls": [{"name": "move_to", "arguments": {"view": best_view, "u": round(u, 2), "v": round(v, 2)}}]
            }
        return oracle_client

    res = run_episode(episode_data=ep, embodiment=emb, config=config, vlm_client_fn=make_oracle_client())
    print(f"Ep {ep_id_str}: Success={res.metrics.success}, SPL={res.metrics.spl:.4f}, NE={res.metrics.navigation_error_m:.2f}m, Actions={res.metrics.simulator_action_count}, Steps={res.total_steps}, Cat={res.failure_category}", flush=True)
    emb.close()
    return res.metrics.success

if __name__ == "__main__":
    for ep_id in ["8628", "1868", "4164", "8627", "3819"]:
        test_ep(ep_id)

