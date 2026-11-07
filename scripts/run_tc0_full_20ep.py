import json, os, math, numpy as np
import habitat_sim
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig, agent_to_camera_frame, reproject_point, 
    pixel_to_normalized, VIEW_AZIMUTHS_DEG
)
from agenticnav_paper_replica.harness.runner import run_episode

def main():
    dev_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"
    with open(dev_path) as f:
        dev_data = json.load(f)
    episodes = dev_data["episodes"]

    config = ReplicaConfig()
    config.max_simulator_actions = 500
    config.max_decision_steps = 25
    config.robot_radius = 0.10
    config.obstacle_point_threshold = 5

    camera_cfg = CameraConfig()

    results = []
    total_rejections = 0
    success_count = 0
    spl_sum = 0.0
    ne_sum = 0.0
    deadlocks = 0

    # Group by scene to avoid reloading scene multiple times
    ep_by_scene = {}
    for ep in episodes:
        scene = ep["scene_id"]
        ep_by_scene.setdefault(scene, []).append(ep)

    for scene_id, scene_eps in ep_by_scene.items():
        print(f"\n================ Loading Scene: {scene_id} ({len(scene_eps)} episodes) ================")
        emb = HabitatGroundEmbodiment(scene_id=scene_id, camera_config=camera_cfg)

        for ep in scene_eps:
            ep_id = str(ep["episode_id"])
            print(f"\n--- Episode {ep_id} ---")

            step_cnt = [0]
            reselect_attempts = [0]
            ep_rejections = [0]

            def make_oracle_client():
                def oracle_client(messages, tools):
                    step_cnt[0] += 1
                    last_msg = messages[-1].get("content", "") if messages else ""
                    is_reselect = isinstance(last_msg, str) and any(
                        w in last_msg.lower() for w in ["reselect", "collision", "rejected", "blocked"]
                    )
                    if is_reselect:
                        reselect_attempts[0] += 1
                        ep_rejections[0] += 1
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

                    # Reselect schedule
                    nudges = [
                        (0.0, 0.0),      # attempt 0: exact target
                        (-0.08, +0.03),  # attempt 1: nudge left, slightly closer
                        (+0.08, +0.03),  # attempt 2: nudge right, slightly closer
                        (-0.15, +0.05),  # attempt 3: wider left
                        (+0.15, +0.05),  # attempt 4: wider right
                    ]
                    nudge_idx = min(reselect_attempts[0], len(nudges) - 1)
                    u_nudge, v_nudge = nudges[nudge_idx]

                    target_d = min(2.0, d_vec)
                    if reselect_attempts[0] > 0:
                        target_d = max(0.40, 1.50 - 0.25 * reselect_attempts[0])

                    if d_vec > target_d:
                        target_pt = pos + (target_d / d_vec) * vec

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
                    if p_c[2] > 0.05:
                        x0, y0, _ = reproject_point(p_c, camera_cfg)
                        u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
                        u = float(np.clip(u + u_nudge, 0.05, 0.95))
                        v_max = 0.98 if (reselect_attempts[0] > 0 or d_vec < 0.60) else 0.95
                        v = float(np.clip(v + v_nudge, 0.52, v_max))
                    else:
                        rel_angle = (theta_deg - VIEW_AZIMUTHS_DEG[best_view] + 180.0) % 360.0 - 180.0
                        u = float(np.clip(0.50 + 0.50 * math.tan(math.radians(rel_angle)) + u_nudge, 0.05, 0.95))
                        v = float(np.clip(0.95 + v_nudge, 0.52, 0.98))

                    return {
                        "thought": "move",
                        "tool_calls": [{
                            "name": "move_to",
                            "arguments": {"view": best_view, "u": round(u, 2), "v": round(v, 2)}
                        }]
                    }
                return oracle_client

            res = run_episode(episode_data=ep, embodiment=emb, config=config, vlm_client_fn=make_oracle_client())
            success_count += int(res.metrics.success)
            spl_sum += res.metrics.spl
            ne_sum += res.metrics.navigation_error_m
            total_rejections += ep_rejections[0]
            if res.failure_category == "safety_deadlock":
                deadlocks += 1

            rec = {
                "episode_id": ep_id,
                "scene_id": ep["scene_id"],
                "success": bool(res.metrics.success),
                "spl": float(res.metrics.spl),
                "navigation_error_m": float(res.metrics.navigation_error_m),
                "actions": int(res.metrics.simulator_action_count),
                "steps": int(res.total_steps),
                "rejections": int(ep_rejections[0]),
                "failure_category": res.failure_category
            }
            results.append(rec)
            print(f"Ep {ep_id}: Success={rec['success']}, SPL={rec['spl']:.4f}, NE={rec['navigation_error_m']:.2f}m, Actions={rec['actions']}, Steps={rec['steps']}, Rejections={rec['rejections']}, Cat={rec['failure_category']}")

        emb.close()

    summary = {
        "num_episodes": len(results),
        "success_rate": round(100.0 * success_count / len(results), 2),
        "spl": round(spl_sum / len(results), 4),
        "mean_navigation_error_m": float(ne_sum / len(results)),
        "total_rejections": total_rejections,
        "deadlocks": deadlocks,
        "episodes": results
    }

    out_file = "audit/metric_validation/tc0_scripted_oracle_20ep.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("FINAL T-C0 20-EPISODE SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("="*60)

if __name__ == "__main__":
    main()

