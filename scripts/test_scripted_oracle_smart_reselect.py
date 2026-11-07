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

    reselect_count = [0]
    def make_oracle_client():
        def oracle_client(messages, tools):
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

            # Look along path for next waypoint
            # When reselecting, take a shorter step along the path to avoid clipping the obstacle corner
            target_distances = [1.50, 1.00, 0.75, 0.50, 2.00]
            target_d = target_distances[min(reselect_count[0], len(target_distances) - 1)]
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
            theta_deg = math.degrees(math.atan2(-x_a, -z_a))

            best_view = 0
            min_diff = 999.0
            for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
                diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
                if diff < min_diff:
                    min_diff = diff
                    best_view = v_idx

            # Angle relative to best_view
            rel_theta_deg = (theta_deg - VIEW_AZIMUTHS_DEG[best_view] + 180.0) % 360.0 - 180.0
            x_pixel = int(round(camera_cfg.cx - camera_cfg.fx * math.tan(math.radians(rel_theta_deg))))
            x_pixel = int(np.clip(x_pixel, 20, camera_cfg.image_width - 21))

            # Find row in depth_views[best_view] whose unprojected distance gives target_d
            d_img = emb._extract_obs().depth_views[best_view]
            col_depth = d_img[:, x_pixel]
            desired_depth = target_d + 0.30

            # Filter valid depth pixels
            valid_y = np.where((col_depth >= camera_cfg.d_min + 0.01) & (col_depth < camera_cfg.d_max - 0.01))[0]
            if len(valid_y) > 0:
                best_y = valid_y[np.argmin(np.abs(col_depth[valid_y] - desired_depth))]
            else:
                best_y = int(camera_cfg.image_height * 0.85)

            u = float(np.clip(x_pixel / camera_cfg.image_width, 0.05, 0.95))
            v = float(np.clip(best_y / camera_cfg.image_height, 0.52, 0.95))

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

