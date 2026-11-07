import os
import sys
import json
import math
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.geometry.pinhole import CameraConfig
import habitat_sim

def run_oracle_to_goal(episodes_path, output_path):
    with open(episodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", data)
    
    camera_cfg = CameraConfig()
    
    # Sort episodes by scene to minimize simulator re-initialization (10 loads total)
    scene_grouped = sorted(episodes, key=lambda e: resolve_scene_path(str(e.get("scene_id", ""))))
    
    results = {}
    current_scene = None
    emb = None
    
    print(f"Running Goal-Reaching Oracle on all {len(episodes)} episodes from {episodes_path}...")
    print("Stop condition: distance to goal <= 0.20m (walk all the way to the goal point)")
    
    for idx, ep in enumerate(scene_grouped, start=1):
        ep_id = str(ep["episode_id"])
        scene_id = ep["scene_id"]
        res_scene = resolve_scene_path(str(scene_id))
        scene_base = res_scene.split("/")[-1].split(".")[0]
        
        if res_scene != current_scene:
            if emb is not None:
                emb.close()
            current_scene = res_scene
            emb = HabitatGroundEmbodiment(
                scene_id=res_scene,
                camera_config=camera_cfg,
                gpu_device_id=-1,
                max_actions=500
            )
            
        obs = emb.reset(ep)
        sim = emb.sim
        pf = sim.pathfinder
        agent = sim.get_agent(0)
        
        start_pos = np.array(ep["start_position"], dtype=np.float32)
        if "goals" in ep and len(ep["goals"]) > 0:
            goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)
        elif "goal_position" in ep:
            goal_pos = np.array(ep["goal_position"], dtype=np.float32)
        elif "reference_path" in ep and len(ep["reference_path"]) > 0:
            goal_pos = np.array(ep["reference_path"][-1], dtype=np.float32)
        else:
            goal_pos = np.array([0.0, 0.0, -10.0], dtype=np.float32)
            
        # Geodesic shortest path
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos
        found = pf.find_path(path)
        
        initial_geodesic = float(path.geodesic_distance) if found else float("nan")
        
        # Calculate reference path length from episode file for comparison
        ref_path_len = 0.0
        if "reference_path" in ep and len(ep["reference_path"]) >= 2:
            rp = [np.array(p, dtype=np.float32) for p in ep["reference_path"]]
            ref_path_len = float(sum(np.linalg.norm(rp[i] - rp[i-1]) for i in range(1, len(rp))))
            
        waypoints = list(path.points) if found else []
        
        # Step through waypoints to within 0.20m of the goal
        cur_pos = start_pos.copy()
        path_travelled = 0.0
        steps_taken = 0
        
        for wp in waypoints[1:]:
            vec = wp - cur_pos
            vec[1] = 0.0
            dist = np.linalg.norm(vec)
            
            while dist > 0.20 and steps_taken < 400:
                dir_vec = vec / dist
                step_len = min(0.25, float(dist))
                step_target = cur_pos + dir_vec * step_len
                stepped = pf.try_step(cur_pos, step_target)
                actual_step = np.linalg.norm(stepped - cur_pos)
                path_travelled += float(actual_step)
                cur_pos = stepped
                steps_taken += 1
                
                # Check distance to goal
                dist_to_goal = np.linalg.norm(cur_pos - goal_pos)
                if dist_to_goal <= 0.20:
                    break
                    
                vec = wp - cur_pos
                vec[1] = 0.0
                dist = np.linalg.norm(vec)
                
            dist_to_goal = np.linalg.norm(cur_pos - goal_pos)
            if dist_to_goal <= 0.20:
                break
                
        final_dist_to_goal = float(np.linalg.norm(cur_pos - goal_pos))
        success = (final_dist_to_goal <= 3.0)
        # Habitat standard SPL calculation
        spl = (initial_geodesic / max(path_travelled, initial_geodesic)) if (success and not math.isnan(initial_geodesic) and initial_geodesic > 0) else 0.0
        
        # Check finite
        assert not math.isinf(final_dist_to_goal), f"Inf distance for ep {ep_id}"
        assert not math.isnan(final_dist_to_goal), f"NaN distance for ep {ep_id}"
        assert not math.isinf(initial_geodesic), f"Inf geodesic for ep {ep_id}"
        
        res = {
            "episode_id": ep_id,
            "scene": scene_base,
            "success": success,
            "spl": spl,
            "initial_geodesic_m": initial_geodesic,
            "reference_path_length_m": ref_path_len,
            "geodesic_diff_m": abs(initial_geodesic - ref_path_len) if ref_path_len > 0 else 0.0,
            "path_travelled_m": path_travelled,
            "final_distance_to_goal_m": final_dist_to_goal,
            "steps_taken": steps_taken,
        }
        results[ep_id] = res
        print(f"[{idx:>3}/100] Ep {ep_id:>4} ({scene_base}): Success={success} | SPL={spl:.4f} | FinalDist={final_dist_to_goal:.2f}m | Path={path_travelled:.2f}m / Geod={initial_geodesic:.2f}m")
        
    if emb is not None:
        emb.close()
        
    # Re-order results to match original episode sequence
    ordered_results = [results[str(e["episode_id"])] for e in episodes]
    
    sr = float(np.mean([r["success"] for r in ordered_results]))
    mean_spl = float(np.mean([r["spl"] for r in ordered_results]))
    mean_ne = float(np.mean([r["final_distance_to_goal_m"] for r in ordered_results]))
    mean_steps = float(np.mean([r["steps_taken"] for r in ordered_results]))
    mean_path = float(np.mean([r["path_travelled_m"] for r in ordered_results]))
    mean_geod = float(np.mean([r["initial_geodesic_m"] for r in ordered_results]))
    
    print("\n" + "="*70)
    print("T-A2 GOAL-REACHING ORACLE RESULTS (ALL 100 EPISODES):")
    print(f"  Episodes evaluated:       {len(ordered_results)} / 100")
    print(f"  Success Rate (SR):        {sr*100:.1f}%")
    print(f"  Mean SPL (Unclamped):     {mean_spl:.4f}")
    print(f"  Mean Final NE:            {mean_ne:.3f} m")
    print(f"  Mean Path Travelled:      {mean_path:.2f} m")
    print(f"  Mean Initial Geodesic:    {mean_geod:.2f} m")
    print(f"  Mean Primitive Steps:     {mean_steps:.1f}")
    print("="*70)
    
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": "T-A2 Goal-Reaching Oracle Follower",
            "episodes_count": len(ordered_results),
            "summary": {
                "success_rate": sr,
                "mean_spl": mean_spl,
                "mean_navigation_error_m": mean_ne,
                "mean_path_travelled_m": mean_path,
                "mean_initial_geodesic_m": mean_geod,
                "mean_steps": mean_steps,
            },
            "episodes": ordered_results
        }, f, indent=2)
    print(f"Saved T-A2 oracle metrics to {output_path}")

if __name__ == "__main__":
    ep_file = "agenticnav_paper_replica/data/opennav_r2rce_100.json"
    out_file = "audit/metric_validation/oracle_to_goal_100.json"
    run_oracle_to_goal(ep_file, out_file)

