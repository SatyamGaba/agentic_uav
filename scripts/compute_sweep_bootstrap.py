"""T-B3 Clustered Bootstrap on Corridor Calibration Data (Vectorized / Fast).

Performs 1,000 pose-level cluster bootstrap iterations across the 30 non-test poses
(1,980 actions total, clustered by 30 physical poses in non-test scenes).
Computes 95% confidence intervals on:
- False Block Rate (FBR) at r_a = 0.18m vs r_a = 0.10m
- False Free Rate (FFR) at r_a = 0.18m vs r_a = 0.10m
- Accuracy at r_a = 0.18m vs r_a = 0.10m
- Relative reduction in False Block Rate: (FBR_0.18 - FBR_0.10) / FBR_0.18
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

from agenticnav_paper_replica.geometry.pinhole import CameraConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from scripts.sweep_g64b_multiscene import extract_all_valid_agent_points
from habitat_sim.utils.common import quat_rotate_vector

dev_scenes = ["gZ6f7yhEvPG", "sT4fr6TAbpF", "JF19kD82Mey"]

def compute_bootstrap():
    camera_cfg = CameraConfig()
    
    headings = [-135, -90, -45, -30, -15, 0, 15, 30, 45, 90, 135]
    distances = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
    
    # Store cluster contingency stats: (tp, tn, fp, fn) per pose cluster
    cluster_stats_18 = []
    cluster_stats_10 = []
    
    print("Collecting 30 pose clusters across 3 non-test scenes...")
    for scene_id in dev_scenes:
        emb = HabitatGroundEmbodiment(
            scene_id=scene_id,
            camera_config=camera_cfg,
            gpu_device_id=-1,
            max_actions=100
        )
        sim = emb.sim
        pf = sim.pathfinder
        
        np.random.seed(1337 + hash(scene_id) % 1000)
        for p_idx in range(10):
            pt = pf.get_random_navigable_point()
            agent = sim.get_agent(0)
            
            agent_state = sim.get_agent(0).get_state()
            agent_state.position = pt
            agent.set_state(agent_state)
            
            obs = emb._extract_obs()
            all_pts = extract_all_valid_agent_points(obs.depth_views, camera_cfg, stride=4)
            if len(all_pts) == 0:
                continue
                
            pos_sim = agent_state.position
            rot_sim = agent_state.rotation
            
            tp18 = tn18 = fp18 = fn18 = 0
            tp10 = tn10 = fp10 = fn10 = 0
            
            for h_deg in headings:
                th_rad = math.radians(h_deg)
                for rho_m in distances:
                    local_disp = np.array([rho_m * np.sin(th_rad), 0.0, -rho_m * np.cos(th_rad)], dtype=np.float32)
                    world_disp = quat_rotate_vector(rot_sim, local_disp)
                    target_pos = pos_sim + world_disp
                    
                    final_pos = pf.try_step(pos_sim, target_pos)
                    dist_moved = float(np.linalg.norm(final_pos - pos_sim))
                    err = float(np.linalg.norm(final_pos - target_pos))
                    
                    truth_blocked = (dist_moved < (rho_m - 0.08)) or (err > 0.08)
                    
                    s = all_pts[:, 0] * np.sin(th_rad) - all_pts[:, 2] * np.cos(th_rad)
                    h = all_pts[:, 0] * np.cos(th_rad) + all_pts[:, 2] * np.sin(th_rad)
                    y = all_pts[:, 1]
                    
                    # Evaluate ra = 0.18
                    mask18 = (y >= 0.15) & (y <= 1.25) & (s >= 0.0) & (s <= (rho_m + 0.18)) & (np.abs(h) <= 0.18)
                    pred_b18 = (int(np.count_nonzero(mask18)) >= 5)
                    if truth_blocked and pred_b18:
                        tp18 += 1
                    elif not truth_blocked and not pred_b18:
                        tn18 += 1
                    elif not truth_blocked and pred_b18:
                        fp18 += 1
                    elif truth_blocked and not pred_b18:
                        fn18 += 1
                        
                    # Evaluate ra = 0.10
                    mask10 = (y >= 0.15) & (y <= 1.25) & (s >= 0.0) & (s <= (rho_m + 0.10)) & (np.abs(h) <= 0.10)
                    pred_b10 = (int(np.count_nonzero(mask10)) >= 5)
                    if truth_blocked and pred_b10:
                        tp10 += 1
                    elif not truth_blocked and not pred_b10:
                        tn10 += 1
                    elif not truth_blocked and pred_b10:
                        fp10 += 1
                    elif truth_blocked and not pred_b10:
                        fn10 += 1
                        
            cluster_stats_18.append((tp18, tn18, fp18, fn18))
            cluster_stats_10.append((tp10, tn10, fp10, fn10))
            
        emb.close()
        print(f"Finished scene {scene_id}, cumulative poses: {len(cluster_stats_18)}")
        
    stats_arr_18 = np.array(cluster_stats_18, dtype=np.int64) # (30, 4)
    stats_arr_10 = np.array(cluster_stats_10, dtype=np.int64) # (30, 4)
    
    num_poses = len(stats_arr_18)
    total_actions = int(np.sum(stats_arr_18))
    print(f"Precomputed {num_poses} pose clusters with {total_actions} total actions.")
    
    # Point estimates on full dataset
    sum18 = np.sum(stats_arr_18, axis=0)
    tp, tn, fp, fn = sum18
    fbr_18_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ffr_18_pt = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc_18_pt = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    sum10 = np.sum(stats_arr_10, axis=0)
    tp, tn, fp, fn = sum10
    fbr_10_pt = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    ffr_10_pt = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    acc_10_pt = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    rel_red_pt = (fbr_18_pt - fbr_10_pt) / fbr_18_pt if fbr_18_pt > 0 else 0.0
    
    # Clustered bootstrap B = 1000
    print("Running 1,000 Pose-Level Cluster Bootstrap Resamples...")
    rng = np.random.RandomState(42)
    B = 1000
    
    boot_fbr_18 = []
    boot_fbr_10 = []
    boot_ffr_18 = []
    boot_ffr_10 = []
    boot_acc_18 = []
    boot_acc_10 = []
    boot_rel_red = []
    
    for b in range(B):
        idx = rng.choice(num_poses, size=num_poses, replace=True)
        s18 = np.sum(stats_arr_18[idx], axis=0)
        s10 = np.sum(stats_arr_10[idx], axis=0)
        
        tp18, tn18, fp18, fn18 = s18
        fbr18 = fp18 / (fp18 + tn18) if (fp18 + tn18) > 0 else 0.0
        ffr18 = fn18 / (fn18 + tp18) if (fn18 + tp18) > 0 else 0.0
        acc18 = (tp18 + tn18) / (tp18 + tn18 + fp18 + fn18)
        
        tp10, tn10, fp10, fn10 = s10
        fbr10 = fp10 / (fp10 + tn10) if (fp10 + tn10) > 0 else 0.0
        ffr10 = fn10 / (fn10 + tp10) if (fn10 + tp10) > 0 else 0.0
        acc10 = (tp10 + tn10) / (tp10 + tn10 + fp10 + fn10)
        
        red = (fbr18 - fbr10) / fbr18 if fbr18 > 0 else 0.0
        
        boot_fbr_18.append(fbr18)
        boot_fbr_10.append(fbr10)
        boot_ffr_18.append(ffr18)
        boot_ffr_10.append(ffr10)
        boot_acc_18.append(acc18)
        boot_acc_10.append(acc10)
        boot_rel_red.append(red)
        
    def ci_95(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        
    report = {
        "task": "T-B3 Pose-Level Clustered Bootstrap on Corridor Calibration",
        "num_poses": num_poses,
        "total_actions": total_actions,
        "bootstrap_iterations": B,
        "results": {
            "ra_018_baseline": {
                "fbr_point": float(fbr_18_pt),
                "fbr_95ci": ci_95(boot_fbr_18),
                "ffr_point": float(ffr_18_pt),
                "ffr_95ci": ci_95(boot_ffr_18),
                "acc_point": float(acc_18_pt),
                "acc_95ci": ci_95(boot_acc_18),
            },
            "ra_010_calibrated": {
                "fbr_point": float(fbr_10_pt),
                "fbr_95ci": ci_95(boot_fbr_10),
                "ffr_point": float(ffr_10_pt),
                "ffr_95ci": ci_95(boot_ffr_10),
                "acc_point": float(acc_10_pt),
                "acc_95ci": ci_95(boot_acc_10),
            },
            "relative_fbr_reduction": {
                "point_estimate": float(rel_red_pt),
                "ci_95": ci_95(boot_rel_red),
            }
        }
    }
    
    out_dir = PROJECT_ROOT / "audit" / "state_integrity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "sweep_cluster_bootstrap.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*80)
    print("T-B3 POSE-LEVEL CLUSTERED BOOTSTRAP RESULTS (B=1000 resamples):")
    print(f"  r_a = 0.18 m Baseline:")
    print(f"    False Block Rate (FBR): {fbr_18_pt*100:.1f}% [95% CI: {np.percentile(boot_fbr_18, 2.5)*100:.1f}%, {np.percentile(boot_fbr_18, 97.5)*100:.1f}%]")
    print(f"    False Free Rate (FFR):  {ffr_18_pt*100:.1f}% [95% CI: {np.percentile(boot_ffr_18, 2.5)*100:.1f}%, {np.percentile(boot_ffr_18, 97.5)*100:.1f}%]")
    print(f"    Overall Accuracy:       {acc_18_pt*100:.1f}% [95% CI: {np.percentile(boot_acc_18, 2.5)*100:.1f}%, {np.percentile(boot_acc_18, 97.5)*100:.1f}%]")
    print(f"  r_a = 0.10 m Calibrated:")
    print(f"    False Block Rate (FBR): {fbr_10_pt*100:.1f}% [95% CI: {np.percentile(boot_fbr_10, 2.5)*100:.1f}%, {np.percentile(boot_fbr_10, 97.5)*100:.1f}%]")
    print(f"    False Free Rate (FFR):  {ffr_10_pt*100:.1f}% [95% CI: {np.percentile(boot_ffr_10, 2.5)*100:.1f}%, {np.percentile(boot_ffr_10, 97.5)*100:.1f}%]")
    print(f"    Overall Accuracy:       {acc_10_pt*100:.1f}% [95% CI: {np.percentile(boot_acc_10, 2.5)*100:.1f}%, {np.percentile(boot_acc_10, 97.5)*100:.1f}%]")
    print(f"  Relative Deadlock-Risk Reduction:")
    print(f"    FBR Reduction:          {rel_red_pt*100:.1f}% [95% CI: {np.percentile(boot_rel_red, 2.5)*100:.1f}%, {np.percentile(boot_rel_red, 97.5)*100:.1f}%]")
    print("="*80)
    print(f"Saved bootstrap report to {out_file}")

if __name__ == "__main__":
    compute_bootstrap()
