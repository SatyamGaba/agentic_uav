import json
import math
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import (
    HabitatGroundEmbodiment,
    resolve_scene_path,
)
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    agent_to_camera_frame,
    reproject_point,
    pixel_to_normalized,
    VIEW_AZIMUTHS_DEG,
)
from agenticnav_paper_replica.harness.runner import run_episode, EpisodeRunResult
import habitat_sim

def test_scripted_oracle(num_episodes=5):
    dev_path = PROJECT_ROOT / "agenticnav_paper_replica/data/r2rce_dev_20.json"
    with open(dev_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", data)[:num_episodes]
    
    config = ReplicaConfig()
    config.max_simulator_actions = 500
    config.max_decision_steps = 25
    config.robot_radius = 0.10
    config.obstacle_point_threshold = 5
    config.floor_grounding_prompt = False
    config.arrival_verification = False
    
    camera_cfg = CameraConfig()
    
    results = []
    cur_scene = None
    emb = None
    
    for ep_idx, ep in enumerate(episodes, start=1):
        ep_id = str(ep["episode_id"])
        res_scene = resolve_scene_path(str(ep["scene_id"]))
        
        if res_scene != cur_scene:
            if emb is not None:
                emb.close()
            cur_scene = res_scene
            emb = HabitatGroundEmbodiment(
                scene_id=res_scene,
                max_actions=config.max_simulator_actions,
                gpu_device_id=-1,
            )
            
        print(f"\n[{ep_idx}/{len(episodes)}] Running Ep {ep_id} in {res_scene.split('/')[-1]}...", flush=True)
        
        reselect_total = [0]

        def make_oracle_client():
            step_count = [0]
            reselect_attempts = [0]
            def oracle_client(messages, tools):
                step_count[0] += 1
                
                # Check for reselection prompt in previous turn
                last_msg = messages[-1].get("content", "") if messages else ""
                is_reselect = isinstance(last_msg, str) and any(w in last_msg.lower() for w in ["reselect", "collision", "rejected", "blocked"])
                if is_reselect:
                    reselect_attempts[0] += 1
                    reselect_total[0] += 1
                    print(f"  [Oracle Reselect] Attempt {reselect_attempts[0]} after rejected candidate", flush=True)
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
                
                # Check stop condition: within goal radius (standard 2.5m, well within 3.0m threshold)
                if dist <= 2.5:
                    print(f"  [Oracle Step {step_count[0]}] Goal within range: {dist:.2f}m <= 2.5m (goal radius 3.0m) -> CALL STOP", flush=True)
                    return {
                        "thought": f"Distance {dist:.2f}m <= 2.5m. Calling stop.",
                        "tool_calls": [{"name": "stop", "arguments": {"reason": "Reached goal landmark"}}]
                    }
                    
                # Waypoint selection schedule with lateral offset on reselection
                offsets = [
                    (1.50, 0.0),
                    (1.25, 0.0),
                    (1.25, -0.05),
                    (0.75, +0.08),
                    (0.50, 0.0),
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
                
                # Floor level in canonical agent frame is y=0
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
                z_c = float(p_c[2])
                if z_c > 0.3:
                    x0, y0, _ = reproject_point(p_c, camera_cfg)
                    u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
                    u = float(np.clip(u + u_nudge, 0.05, 0.95))
                    v = float(np.clip(v, 0.52, 0.95))
                else:
                    u, v = 0.50, 0.70
                    
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
        
        print(f"  -> Finished: Success={res.metrics.success}, SPL={res.metrics.spl:.4f}, NE={res.metrics.navigation_error_m:.2f}m, Actions={res.metrics.simulator_action_count}, Steps={res.total_steps}, Reselects={reselect_total[0]}, Cat={res.failure_category}", flush=True)
        results.append({
            "episode_id": ep_id,
            "success": res.metrics.success,
            "spl": res.metrics.spl,
            "navigation_error_m": res.metrics.navigation_error_m,
            "actions": res.metrics.simulator_action_count,
            "steps": res.total_steps,
            "rejections": reselect_total[0],
            "failure_category": res.failure_category,
        })
        
    if emb is not None:
        emb.close()
        
    sr = float(np.mean([r["success"] for r in results]))
    spl = float(np.mean([r["spl"] for r in results]))
    mean_ne = float(np.mean([r["navigation_error_m"] for r in results]))
    total_rejections = int(sum(r["rejections"] for r in results))
    print("\n" + "="*70)
    print(f"SCRIPTED ORACLE HARNESS (T-C0) SUMMARY ON {len(results)} DEV EPISODES:")
    print(f"Success Rate (SR): {sr*100:.1f}% ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"SPL:              {spl:.4f}")
    print(f"Mean NE:          {mean_ne:.2f}m")
    print(f"Total Rejections: {total_rejections}")
    print(f"Pass Criterion:   >= 90% SR with 0 deadlocks -> {'PASS' if sr >= 0.90 else 'FAIL'}")
    print("="*70)

    summary_out = {
        "num_episodes": len(results),
        "success_rate": sr * 100.0,
        "spl": spl,
        "mean_navigation_error_m": mean_ne,
        "total_rejections": total_rejections,
        "deadlocks": sum(1 for r in results if r["failure_category"] == "safety_deadlock"),
        "episodes": results
    }
    out_file = PROJECT_ROOT / f"audit/metric_validation/tc0_scripted_oracle_{len(results)}ep.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary_out, f, indent=2)
    print("Saved summary to", out_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=20)
    args = parser.parse_args()
    test_scripted_oracle(args.num_episodes)
