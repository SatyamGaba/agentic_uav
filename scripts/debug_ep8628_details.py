import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, agent_to_camera_frame, reproject_point, pixel_to_normalized, VIEW_AZIMUTHS_DEG, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.harness.runner import run_episode

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

config = ReplicaConfig()
config.max_simulator_actions = 500
config.max_decision_steps = 25
config.robot_radius = 0.10
config.obstacle_point_threshold = 5
config.floor_grounding_prompt = False
config.arrival_verification = False

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

        offsets = [(1.50, 0.0), (1.25, 0.0), (1.25, -0.05), (0.75, +0.08), (0.50, 0.0)]
        idx = min(reselect_attempts[0], len(offsets) - 1)
        target_d, u_nudge = offsets[idx]
        target_d = min(target_d, dist)

        pts = list(path.points)
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
            u = float(np.clip(u + u_nudge, 0.05, 0.95))
            v = float(np.clip(v, 0.52, 0.95))
        else:
            u, v = 0.50, 0.70

        return {
            "thought": "move",
            "tool_calls": [{"name": "move_to", "arguments": {"view": best_view, "u": round(u, 2), "v": round(v, 2)}}]
        }
    return oracle_client

res = run_episode(episode_data=ep, embodiment=emb, config=config, vlm_client_fn=make_oracle_client())
emb.close()

if res.trace:
    for step in res.trace.steps:
        s_idx = step.node_idx
        act = step.action_type
        p_dist = step.geodesic_distance_to_goal_m
        outcome = step.outcome or {}
        payload = step.action_payload or {}
        turns = payload.get("turn_steps")
        fwds = payload.get("forward_steps")
        moved = outcome.get("moved_m")
        collided = outcome.get("collided")
        print(f"Step {s_idx}: act={act}, turns={turns}, fwds={fwds}, moved={moved}, col={collided}, dist={p_dist:.2f}")
else:
    print("Metrics:", res.metrics, "Category:", res.failure_category)
