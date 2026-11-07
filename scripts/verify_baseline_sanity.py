"""
Verification Script: Oracle & Baseline Sanity Battery (scripts/verify_baseline_sanity.py)

Tests:
1. Shortest-Path Oracle Follower: Steps along the pathfinder shortest path and calls stop() at the goal.
   Verifies whether SR ~ 100% and SPL ~ 1.0 (verifies metric computation integrity).
2. Stop-Immediately Agent: Calls stop() at step 1. Measures start-to-goal geodesic distance.
3. Random Walker Agent: Takes uniform random discrete navigation actions for budget.
4. Start-to-Goal Geodesic Distance Ledger for all 15 episodes.
"""

import sys
import os
import json
import math
import random
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.evaluation.metrics import compute_episode_metrics, EpisodeMetrics
import habitat_sim

def run_oracle_shortest_path_follower(embodiment: HabitatGroundEmbodiment, ep_data: dict, max_actions: int = 50):
    obs = embodiment.reset(ep_data)
    pf = embodiment.sim.pathfinder
    start_pos = np.array(ep_data.get("start_position", [0.0, 0.0, 0.0]), dtype=np.float32)
    goal_pos = np.array(embodiment.current_goal_pos, dtype=np.float32)
    
    path = habitat_sim.ShortestPath()
    path.requested_start = start_pos
    path.requested_end = goal_pos
    found = pf.find_path(path)
    if not found or len(path.points) < 2:
        # Fallback straight line
        waypoints = [start_pos, goal_pos]
    else:
        waypoints = path.points

    trajectory = [obs.allocentric_pose]
    geodesic_dists = [embodiment.get_geodesic_distance_to_goal()]
    
    target_idx = 1
    actions_taken = 0
    called_stop = False
    
    while actions_taken < max_actions:
        dist_to_goal = embodiment.get_geodesic_distance_to_goal()
        if dist_to_goal <= 1.5:  # safely inside 3.0m threshold
            called_stop = True
            break
            
        cur_state = embodiment.sim.get_agent(0).get_state()
        cur_pos = np.array(cur_state.position)
        target_pt = waypoints[min(target_idx, len(waypoints) - 1)]
        
        # Vector to target waypoint in horizontal plane (X, Z)
        dx = target_pt[0] - cur_pos[0]
        dz = target_pt[2] - cur_pos[2]
        dist_wp = math.sqrt(dx*dx + dz*dz)
        
        if dist_wp < 0.30 and target_idx < len(waypoints) - 1:
            target_idx += 1
            continue
            
        # Target heading
        target_yaw = math.atan2(-dx, -dz)
        cur_yaw = obs.allocentric_pose[2]
        
        yaw_diff = (target_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_diff_deg = math.degrees(yaw_diff)
        
        if abs(yaw_diff_deg) > 15.0:
            turn_steps = 1 if yaw_diff_deg > 0 else -1
            forward_steps = 0
        else:
            turn_steps = 0
            forward_steps = 1
            
        obs = embodiment.step_discrete(turn_steps_15deg=turn_steps, forward_steps_025m=forward_steps)
        actions_taken += abs(turn_steps) + forward_steps
        if obs.primitive_trajectory:
            trajectory.extend(obs.primitive_trajectory)
            geodesic_dists.extend(obs.primitive_geodesic_distances)
        else:
            trajectory.append(obs.allocentric_pose)
            geodesic_dists.append(embodiment.get_geodesic_distance_to_goal())
            
        if obs.collided and forward_steps > 0:
            # Advance waypoint if stuck
            target_idx = min(target_idx + 1, len(waypoints) - 1)

    # Final arrival stop
    if not called_stop and embodiment.get_geodesic_distance_to_goal() <= 3.0:
        called_stop = True

    metrics = compute_episode_metrics(
        trajectory=trajectory,
        distances_to_goal=geodesic_dists,
        shortest_path_length=embodiment.get_shortest_path_distance(),
        simulator_action_count=actions_taken,
        called_stop=called_stop,
        success_distance_threshold=3.0,
    )
    return metrics, len(trajectory)

def run_stop_immediately(embodiment: HabitatGroundEmbodiment, ep_data: dict):
    obs = embodiment.reset(ep_data)
    start_dist = embodiment.get_geodesic_distance_to_goal()
    metrics = compute_episode_metrics(
        trajectory=[obs.allocentric_pose],
        distances_to_goal=[start_dist],
        shortest_path_length=embodiment.get_shortest_path_distance(),
        simulator_action_count=0,
        called_stop=True,
        success_distance_threshold=3.0,
    )
    return metrics, start_dist

def run_random_walker(embodiment: HabitatGroundEmbodiment, ep_data: dict, max_actions: int = 25, seed: int = 42):
    random.seed(seed)
    obs = embodiment.reset(ep_data)
    trajectory = [obs.allocentric_pose]
    geodesic_dists = [embodiment.get_geodesic_distance_to_goal()]
    actions_taken = 0
    called_stop = False
    
    for _ in range(max_actions):
        choice = random.choice(["fwd", "left", "right", "fwd"])
        if choice == "fwd":
            obs = embodiment.step_discrete(0, 1)
            actions_taken += 1
        elif choice == "left":
            obs = embodiment.step_discrete(1, 0)
            actions_taken += 1
        elif choice == "right":
            obs = embodiment.step_discrete(-1, 0)
            actions_taken += 1
            
        if obs.primitive_trajectory:
            trajectory.extend(obs.primitive_trajectory)
            geodesic_dists.extend(obs.primitive_geodesic_distances)
        else:
            trajectory.append(obs.allocentric_pose)
            geodesic_dists.append(embodiment.get_geodesic_distance_to_goal())
            
        if embodiment.get_geodesic_distance_to_goal() <= 1.0:
            called_stop = True
            break
            
    metrics = compute_episode_metrics(
        trajectory=trajectory,
        distances_to_goal=geodesic_dists,
        shortest_path_length=embodiment.get_shortest_path_distance(),
        simulator_action_count=actions_taken,
        called_stop=called_stop,
        success_distance_threshold=3.0,
    )
    return metrics

def main():
    print("=" * 80)
    print("ORACLE & BASELINE SANITY BATTERY (Check G9.2)")
    print("=" * 80)
    
    ep_file = PROJECT_ROOT / "agenticnav_paper_replica/data/episode_list_100.json"
    with open(ep_file) as f:
        all_eps = json.load(f)["episodes"][:15]
        
    scene_id = "gZ6f7yhEvPG"
    print(f"Initializing HabitatGroundEmbodiment on {scene_id}...")
    embodiment = HabitatGroundEmbodiment(scene_id=scene_id, gpu_device_id=-1, max_actions=50)
    
    oracle_results = []
    stop_imm_results = []
    random_results = []
    
    print("\nRunning across 15 episodes:")
    print(f"{'Idx':<4} {'Episode ID':<24} {'StartDist':<10} {'Oracle SR':<10} {'Oracle SPL':<11} {'StopImm SR':<11} {'RandWalker OSR':<14} {'RandWalker SR'}")
    print("-" * 95)
    
    for idx, ep in enumerate(all_eps):
        ep_id = ep["episode_id"]
        ep_copy = dict(ep)
        ep_copy["scene_id"] = scene_id
        
        # 1. Oracle Shortest Path Follower
        m_oracle, _ = run_oracle_shortest_path_follower(embodiment, ep_copy)
        oracle_results.append(m_oracle)
        
        # 2. Stop Immediately
        m_stop, start_dist = run_stop_immediately(embodiment, ep_copy)
        stop_imm_results.append(m_stop)
        
        # 3. Random Walker
        m_rand = run_random_walker(embodiment, ep_copy, seed=42 + idx)
        random_results.append(m_rand)
        
        print(f"{idx+1:<4} {ep_id:<24} {start_dist:<10.2f} {str(m_oracle.success):<10} {m_oracle.spl:<11.3f} {str(m_stop.success):<11} {str(m_rand.oracle_success):<14} {str(m_rand.success)}")
        
    embodiment.close()
    
    # Aggregates
    oracle_sr = np.mean([1.0 if m.success else 0.0 for m in oracle_results]) * 100.0
    oracle_spl = np.mean([m.spl for m in oracle_results]) * 100.0
    
    stop_sr = np.mean([1.0 if m.success else 0.0 for m in stop_imm_results]) * 100.0
    
    rand_osr = np.mean([1.0 if m.oracle_success else 0.0 for m in random_results]) * 100.0
    rand_sr = np.mean([1.0 if m.success else 0.0 for m in random_results]) * 100.0
    rand_ne = np.mean([m.navigation_error_m for m in random_results])
    
    print("\n" + "=" * 80)
    print("SANITY BATTERY SUMMARY RESULTS:")
    print("=" * 80)
    print(f"1. Shortest-Path Follower : SR = {oracle_sr:.1f}% | SPL = {oracle_spl:.1f}%")
    print(f"2. Stop-Immediately Agent : SR = {stop_sr:.1f}% (episodes starting <= 3.0m from goal)")
    print(f"3. Random Walker (25 act) : OSR = {rand_osr:.1f}% | SR = {rand_sr:.1f}% | Mean NE = {rand_ne:.2f}m")
    print("=" * 80)
    
    out_data = {
        "shortest_path_follower": {"sr_pct": oracle_sr, "spl_pct": oracle_spl},
        "stop_immediately": {"sr_pct": stop_sr},
        "random_walker": {"osr_pct": rand_osr, "sr_pct": rand_sr, "mean_ne_m": rand_ne},
        "episodes": [
            {
                "idx": i + 1,
                "episode_id": all_eps[i]["episode_id"],
                "start_distance_m": round(stop_imm_results[i].navigation_error_m, 2),
                "oracle_follower": {"sr": oracle_results[i].success, "spl": round(oracle_results[i].spl, 3), "ne": round(oracle_results[i].navigation_error_m, 2)},
                "stop_immediately": {"sr": stop_imm_results[i].success},
                "random_walker": {"osr": random_results[i].oracle_success, "sr": random_results[i].success, "ne": round(random_results[i].navigation_error_m, 2)},
            }
            for i in range(len(all_eps))
        ]
    }
    
    out_file = PROJECT_ROOT / "audit" / "phase9" / "baseline_sanity_battery.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"Sanity results written to {out_file}")

if __name__ == "__main__":
    main()
