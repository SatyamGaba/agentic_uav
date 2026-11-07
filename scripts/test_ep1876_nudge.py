import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json, numpy as np, math, habitat_sim
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    agent_to_camera_frame,
    reproject_point,
    pixel_to_normalized,
    VIEW_AZIMUTHS_DEG,
)
from agenticnav_paper_replica.harness.runner import run_episode

dev_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"
with open(dev_path) as f:
    ep = json.load(f)["episodes"][3] # Ep 1876

config = ReplicaConfig()
config.max_simulator_actions = 500
config.max_decision_steps = 25
config.robot_radius = 0.10
config.obstacle_point_threshold = 5
config.floor_grounding_prompt = False
config.arrival_verification = False

emb = HabitatGroundEmbodiment(scene_id=resolve_scene_path(ep["scene_id"]), max_actions=500, gpu_device_id=-1)
camera_cfg = CameraConfig()

reselect_attempts = [0]
def make_oracle_client():
    step_count = [0]
    def oracle_client(messages, tools):
        step_count[0] += 1
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
            print(f"  [Oracle Step {step_count[0]}] Goal within range: {dist:.2f}m <= 2.5m (goal radius 3.0m) -> CALL STOP", flush=True)
            return {
                "thought": f"Distance {dist:.2f}m <= 2.5m. Calling stop.",
                "tool_calls": [{"name": "stop", "arguments": {"reason": "Reached goal landmark"}}]
            }
            
        # Select target distance and lateral offset based on reselect attempts
        offsets = [
            (1.5, 0.0),
            (1.25, -0.05),
            (1.25, +0.05),
            (1.0, -0.08),
            (0.75, +0.08),
        ]
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
        x0, y0, _ = reproject_point(p_c, camera_cfg)
        u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
        u = float(np.clip(u + u_nudge, 0.05, 0.95))
        v = float(np.clip(v, 0.52, 0.95))
        
        print(f"  [Oracle Step {step_count[0]}] dist={dist:.2f}m, target_d={target_d:.2f}m, reselect={reselect_attempts[0]}, heading={theta_deg:.1f}deg -> view={best_view}, u={u:.2f}, v={v:.2f}", flush=True)
        return {
            "thought": f"Navigating along shortest path (dist={dist:.2f}m)",
            "tool_calls": [{
                "name": "move_to",
                "arguments": {
                    "view": best_view,
                    "u": round(u, 2),
                    "v": round(v, 2),
                    "reason": f"Follow shortest path, remaining={dist:.2f}m"
                }
            }]
        }
    return oracle_client

res = run_episode(
    episode_data=ep,
    embodiment=emb,
    config=config,
    vlm_client_fn=make_oracle_client(),
)
print(f"Result on Ep 1876: Success={res.metrics.success}, SPL={res.metrics.spl:.4f}, NE={res.metrics.navigation_error_m:.2f}m, Actions={res.metrics.simulator_action_count}, Cat={res.failure_category}")
emb.close()
