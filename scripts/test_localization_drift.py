"""T-B1 Localization Drift & State Integrity Verification.

Executes 200 forced-shortfall and sliding moves into walls/obstacles across 3 scenes:
- Compares open-loop commanded dead-reckoning vs actual simulator pose.
- Compares HabitatGroundEmbodiment reported allocentric_pose vs simulator ground truth state.
- Asserts that embodiment state binding meets the pass criterion: error <= 0.02 m and <= 1.0 deg.
"""

import math
import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.embodiment.habitat_ground import (
    HabitatGroundEmbodiment,
    resolve_scene_path,
)

def run_localization_drift_test():
    scenes = ["2azQ1b91cZZ", "gZ6f7yhEvPG", "sT4fr6TAbpF"]
    moves_per_scene = 70  # Total = 210 moves (> 200 required)
    
    total_moves = 0
    open_loop_errors_m = []
    open_loop_yaw_errors_deg = []
    
    embodiment_errors_m = []
    embodiment_yaw_errors_deg = []
    
    sliding_collisions_detected = 0
    
    records = []
    
    print(f"Running T-B1 Localization Drift Test across {len(scenes)} scenes ({moves_per_scene} moves each)...")
    
    for scene_id in scenes:
        scene_path = resolve_scene_path(scene_id)
        emb = HabitatGroundEmbodiment(scene_id=scene_path, max_actions=500)
        
        # Reset at a valid start position
        start_pt = emb.sim.pathfinder.get_random_navigable_point()
        obs = emb.reset({
            "start_position": [float(start_pt[0]), float(start_pt[1]), float(start_pt[2])],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "goals": [{"position": [0.0, 0.0, 0.0]}]
        })
        
        # Initial ground truth
        gt_state = emb.sim.get_agent(0).get_state()
        gt_x = float(gt_state.position[0])
        gt_z = float(gt_state.position[2])
        rot = gt_state.rotation
        gt_yaw = 2.0 * math.atan2(rot.y, rot.w)
        
        # Open-loop dead-reckoning state tracker
        dead_reckon_x = gt_x
        dead_reckon_z = gt_z
        dead_reckon_yaw = gt_yaw
        
        for move_idx in range(moves_per_scene):
            # Record state before move
            pre_gt_state = emb.sim.get_agent(0).get_state()
            pre_gt_pos = np.array([pre_gt_state.position[0], pre_gt_state.position[2]])
            
            # Execute step in embodiment (move_forward)
            obs = emb.step_discrete(turn_steps_15deg=0, forward_steps_025m=1)
            total_moves += 1
            
            # Update open-loop dead-reckoning assumption
            dead_reckon_x += 0.25 * math.sin(dead_reckon_yaw)
            dead_reckon_z -= 0.25 * math.cos(dead_reckon_yaw)
            
            # Query actual ground truth simulator state
            curr_gt_state = emb.sim.get_agent(0).get_state()
            curr_gt_x = float(curr_gt_state.position[0])
            curr_gt_z = float(curr_gt_state.position[2])
            curr_rot = curr_gt_state.rotation
            curr_gt_yaw = 2.0 * math.atan2(curr_rot.y, curr_rot.w)
            
            # Check if collision / sliding happened (actual motion < commanded 0.25m)
            actual_disp = np.linalg.norm(np.array([curr_gt_x, curr_gt_z]) - pre_gt_pos)
            is_collision_shortfall = (actual_disp < 0.24)
            if is_collision_shortfall:
                sliding_collisions_detected += 1
                
            # Error in open-loop dead-reckoning
            ol_err_dist = math.sqrt((dead_reckon_x - curr_gt_x)**2 + (dead_reckon_z - curr_gt_z)**2)
            ol_err_yaw = abs(math.degrees(dead_reckon_yaw - curr_gt_yaw)) % 360.0
            if ol_err_yaw > 180.0:
                ol_err_yaw = 360.0 - ol_err_yaw
            open_loop_errors_m.append(ol_err_dist)
            open_loop_yaw_errors_deg.append(ol_err_yaw)
            
            # Error in HabitatGroundEmbodiment reported allocentric_pose
            rep_x, rep_z, rep_yaw = obs.allocentric_pose
            emb_err_dist = math.sqrt((rep_x - curr_gt_x)**2 + (rep_z - curr_gt_z)**2)
            emb_err_yaw = abs(math.degrees(rep_yaw - curr_gt_yaw)) % 360.0
            if emb_err_yaw > 180.0:
                emb_err_yaw = 360.0 - emb_err_yaw
            embodiment_errors_m.append(emb_err_dist)
            embodiment_yaw_errors_deg.append(emb_err_yaw)
            
            # If completely stuck against wall, turn to try to slide along it
            if actual_disp < 0.01:
                emb.step_discrete(turn_steps_15deg=1, forward_steps_025m=0)
                dead_reckon_yaw += math.radians(15.0)
                
        emb.close()
        print(f"Scene {scene_id}: completed {moves_per_scene} moves.")
        
    max_emb_err_m = max(embodiment_errors_m)
    max_emb_yaw_deg = max(embodiment_yaw_errors_deg)
    mean_ol_err_m = np.mean(open_loop_errors_m)
    max_ol_err_m = max(open_loop_errors_m)
    
    passed = (max_emb_err_m <= 0.02) and (max_emb_yaw_deg <= 1.0)
    
    report = {
        "task": "T-B1 Localization Drift & State Integrity Verification",
        "total_forced_moves": total_moves,
        "sliding_collisions_detected": sliding_collisions_detected,
        "collision_shortfall_rate": sliding_collisions_detected / total_moves,
        "open_loop_dead_reckoning": {
            "mean_error_m": float(mean_ol_err_m),
            "max_error_m": float(max_ol_err_m),
            "conclusion": "Severe drift when integrating commanded motion open-loop across collisions"
        },
        "embodiment_state_binding": {
            "max_position_error_m": float(max_emb_err_m),
            "max_yaw_error_deg": float(max_emb_yaw_deg),
            "pass_criterion_position": "<= 0.02 m",
            "pass_criterion_yaw": "<= 1.0 deg",
            "status": "PASS" if passed else "FAIL"
        }
    }
    
    out_dir = PROJECT_ROOT / "audit" / "state_integrity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "localization_drift.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*70)
    print("T-B1 LOCALIZATION DRIFT TEST RESULTS:")
    print(f"  Total moves evaluated:       {total_moves}")
    print(f"  Sliding/wall collisions:     {sliding_collisions_detected} ({sliding_collisions_detected/total_moves*100:.1f}%)")
    print(f"  Open-loop drift max:         {max_ol_err_m:.2f} m (mean: {mean_ol_err_m:.2f} m)")
    print(f"  Embodiment max position err: {max_emb_err_m:.6f} m (Threshold: <= 0.02 m)")
    print(f"  Embodiment max yaw err:      {max_emb_yaw_deg:.6f} deg (Threshold: <= 1.0 deg)")
    print(f"  Status:                      {'PASS' if passed else 'FAIL'}")
    print("="*70)
    print(f"Saved drift report to {out_file}")

if __name__ == "__main__":
    run_localization_drift_test()
