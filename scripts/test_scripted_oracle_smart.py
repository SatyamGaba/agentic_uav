import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, agent_to_camera_frame, reproject_point, pixel_to_normalized, VIEW_AZIMUTHS_DEG
from agenticnav_paper_replica.harness.runner import run_episode

def test_episode_smart(ep_id_str):
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
        step_cnt = [0]
        reselect_attempts = [0]
        def oracle_client(messages, tools):
            step_cnt[0] += 1
            last_msg = messages[-1].get("content", "") if messages else ""
            is_reselect = isinstance(last_msg, str) and any(w in last_msg.lower() for w in ["reselect", "collision", "rejected", "blocked"])
            if is_reselect:
                reselect_attempts[0] += 1
                print(f"  [Step {step_cnt[0]} Reselect {reselect_attempts[0]}] Reason: {last_msg}")
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
                print(f"  [Step {step_cnt[0]}] dist={dist:.2f} <= 2.5m -> STOP")
                return {"thought": "stop", "tool_calls": [{"name": "stop", "arguments": {"reason": "Reached"}}]}

            pts = list(path.points)
            
            # Line-of-sight pure pursuit:
            # Find the furthest waypoint along path that has direct line of sight from pos
            furthest_los_idx = 1
            for i in range(1, len(pts)):
                p_test = habitat_sim.ShortestPath()
                p_test.requested_start = pos
                p_test.requested_end = pts[i]
                pf.find_path(p_test)
                if len(p_test.points) == 2:
                    furthest_los_idx = i
                else:
                    break

            target_pt = pts[furthest_los_idx]
            vec = target_pt - pos
            d_vec = float(np.linalg.norm(vec))

            # If the segment is long (> 2.0m), cap the target distance to 2.0m
            # If reselecting, shorten target distance
            max_d = 2.0
            if reselect_attempts[0] > 0:
                max_d = max(0.50, 1.50 - 0.30 * reselect_attempts[0])

            if d_vec > max_d:
                target_pt = pos + (max_d / d_vec) * vec
            elif d_vec < 0.35 and len(pts) > 2 and furthest_los_idx == 1:
                # If target is very close (< 35cm) and ahead, we can step through it slightly along the ray
                target_pt = pos + (0.45 / max(d_vec, 1e-4)) * vec

            dx = float(target_pt[0] - pos[0])
            dz = float(target_pt[2] - pos[2])
            cos_y = math.cos(yaw)
            sin_y = math.sin(yaw)
            x_a = cos_y * dx - sin_y * dz
            z_a = sin_y * dx + cos_y * dz
            p_agent = np.array([x_a, 0.0, z_a], dtype=np.float64)
            theta_rad = math.atan2(-x_a, -z_a)
            theta_deg = math.degrees(theta_rad)

            best_view = 0
            min_diff = 999.0
            for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
                diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
                if diff < min_diff:
                    min_diff = diff
                    best_view = v_idx

            p_c = agent_to_camera_frame(p_agent, best_view, camera_cfg)
            if p_c[2] > 0.3:
                x0, y0, _ = reproject_point(p_c, camera_cfg)
                u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
                u = float(np.clip(u, 0.05, 0.95))
                v = float(np.clip(v, 0.52, 0.95))
            else:
                u, v = 0.50, 0.70

            print(f"  [Step {step_cnt[0]}] dist={dist:.2f}m, los_wp={furthest_los_idx}/{len(pts)-1}, d_vec={d_vec:.2f}m -> view={best_view}, u={u:.2f}, v={v:.2f}")
            return {
                "thought": "move",
                "tool_calls": [{
                    "name": "move_to",
                    "arguments": {"view": best_view, "u": round(u, 2), "v": round(v, 2)}
                }]
            }
        return oracle_client

    res = run_episode(episode_data=ep, embodiment=emb, config=config, vlm_client_fn=make_oracle_client())
    print(f"Ep {ep_id_str}: Success={res.metrics.success}, SPL={res.metrics.spl:.4f}, NE={res.metrics.navigation_error_m:.2f}m, Actions={res.metrics.simulator_action_count}, Steps={res.total_steps}, Cat={res.failure_category}")
    emb.close()
    return res.metrics.success

if __name__ == "__main__":
    import sys
    ep_id = sys.argv[1] if len(sys.argv) > 1 else "8628"
    test_episode_smart(ep_id)
