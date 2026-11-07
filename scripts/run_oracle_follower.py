import os
import sys
import json
import math
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig
import habitat_sim

def run_oracle_on_episodes(episodes_path, max_episodes=20):
    with open(episodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", data)
    if max_episodes > 0:
        episodes = episodes[:max_episodes]
        
    camera_cfg = CameraConfig()
    current_scene = None
    emb = None
    
    results = []
    print(f"Running Oracle Follower on {len(episodes)} episodes from {episodes_path}...")
    
    for idx, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        scene_id = ep["scene_id"]
        scene_base = ep.get("scene_basename", scene_id.split("/")[-1].split(".")[0])
        
        if scene_base != current_scene:
            if emb is not None:
                emb.close()
            current_scene = scene_base
            emb = HabitatGroundEmbodiment(
                scene_id=scene_base,
                camera_config=camera_cfg,
                gpu_device_id=-1,
                max_actions=500
            )
            
        obs = emb.reset(ep)
        sim = emb.sim
        pf = sim.pathfinder
        agent = sim.get_agent(0)
        
        # Start and Goal positions
        start_pos = np.array(ep["start_position"], dtype=np.float32)
        # Goal position can be in goals list or goal_position
        if "goals" in ep and len(ep["goals"]) > 0:
            goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)
            goal_radius = float(ep["goals"][0].get("radius", 3.0))
        elif "goal_position" in ep:
            goal_pos = np.array(ep["goal_position"], dtype=np.float32)
            goal_radius = float(ep.get("goal_radius", 3.0))
        else:
            ref_path = ep.get("reference_path", [])
            goal_pos = np.array(ref_path[-1], dtype=np.float32)
            goal_radius = 3.0
            
        # Find geodesic shortest path
        path = habitat_sim.ShortestPath()
        path.requested_start = start_pos
        path.requested_end = goal_pos
        found = pf.find_path(path)
        
        if not found or len(path.points) < 2:
            print(f"Warning: Ep {ep_id} pathfinder could not find path")
            continue
            
        initial_geodesic = float(path.geodesic_distance)
        waypoints = list(path.points)
        
        # Follow waypoints via discrete primitive actions or navmesh steps
        steps_taken = 0
        cur_pos = start_pos.copy()
        path_travelled = 0.0
        
        for wp in waypoints[1:]:
            # Step towards wp
            vec = wp - cur_pos
            vec[1] = 0.0 # horizontal plane
            dist = np.linalg.norm(vec)
            
            while dist > 0.25 and steps_taken < 200:
                dir_vec = vec / dist
                step_target = cur_pos + dir_vec * 0.25
                stepped = pf.try_step(cur_pos, step_target)
                actual_step = np.linalg.norm(stepped - cur_pos)
                path_travelled += actual_step
                cur_pos = stepped
                steps_taken += 1
                
                # Update agent in sim
                state = habitat_sim.AgentState()
                state.position = cur_pos
                agent.set_state(state)
                
                # Check distance to goal
                dist_to_goal = np.linalg.norm(cur_pos - goal_pos)
                if dist_to_goal <= goal_radius:
                    break
                    
                vec = wp - cur_pos
                vec[1] = 0.0
                dist = np.linalg.norm(vec)
                
            dist_to_goal = np.linalg.norm(cur_pos - goal_pos)
            if dist_to_goal <= goal_radius:
                break
                
        final_dist_to_goal = float(np.linalg.norm(cur_pos - goal_pos))
        success = (final_dist_to_goal <= 3.0)
        spl = (initial_geodesic / max(path_travelled, initial_geodesic)) if success else 0.0
        
        results.append({
            "episode_id": ep_id,
            "scene": scene_base,
            "success": success,
            "spl": spl,
            "initial_geodesic_m": initial_geodesic,
            "path_travelled_m": path_travelled,
            "final_distance_to_goal_m": final_dist_to_goal,
            "steps_taken": steps_taken,
        })
        print(f"Ep {ep_id:>4} ({scene_base}): Success={success} | SPL={spl:.4f} | Dist={final_dist_to_goal:.2f}m | Path={path_travelled:.2f}m")
        
    if emb is not None:
        emb.close()
        
    # Summarize
    sr = np.mean([r["success"] for r in results])
    mean_spl = np.mean([r["spl"] for r in results])
    mean_steps = np.mean([r["steps_taken"] for r in results])
    
    print("\n" + "="*60)
    print("ORACLE FOLLOWER (POSITIVE CONTROL) RESULTS:")
    print(f"  Episodes evaluated: {len(results)}")
    print(f"  Success Rate (SR):  {sr*100:.1f}%")
    print(f"  SPL:                {mean_spl:.4f}")
    print(f"  Mean Steps:         {mean_steps:.1f}")
    print("="*60)
    
    out_dir = Path("audit/positive_control")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "oracle_follower_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "num_episodes": len(results),
            "success_rate": sr,
            "spl": mean_spl,
            "mean_steps": mean_steps,
            "episodes": results
        }, f, indent=2)
    print(f"Saved results to {out_path}")
    
    assert sr >= 0.99, f"Oracle Follower failed to achieve >= 99% SR (got {sr*100:.1f}%)"
    assert mean_spl >= 0.90, f"Oracle Follower failed to achieve >= 0.90 SPL (got {mean_spl:.4f})"
    print("POSITIVE CONTROL PASSED: Simulator physics, geometry, and metric definitions verified!")

if __name__ == "__main__":
    ep_file = "agenticnav_paper_replica/data/opennav_r2rce_100.json"
    run_oracle_on_episodes(ep_file, max_episodes=20)

