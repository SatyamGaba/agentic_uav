import os
import sys
import json
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agenticnav_paper_replica.geometry.pinhole import CameraConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from scripts.sweep_g64b_multiscene import extract_all_valid_agent_points
from habitat_sim.utils.common import quat_rotate_vector

# Independent non-test training/dev scenes (zero overlap with val_unseen test set)
dev_scenes = ["gZ6f7yhEvPG", "sT4fr6TAbpF", "JF19kD82Mey"]

def run_independent_sweep():
    camera_cfg = CameraConfig()
    
    # Grid of candidate parameters
    r_a_grid = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
    y_min_grid = [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    y_max = 1.25
    n_thresh_grid = [1, 3, 5, 10]
    
    headings = [-135, -90, -45, -30, -15, 0, 15, 30, 45, 90, 135] # 11 headings
    distances = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5] # 6 distances
    
    evaluated_actions = [] # list of dicts: {"truth_blocked": bool, "pts": ndarray}
    
    print(f"Calibrating on {len(dev_scenes)} non-test scenes: {dev_scenes}")
    
    for scene_id in dev_scenes:
        emb = HabitatGroundEmbodiment(
            scene_id=scene_id,
            camera_config=camera_cfg,
            gpu_device_id=-1,
            max_actions=100
        )
        sim = emb.sim
        pf = sim.pathfinder
        
        # Sample 10 random valid navigable points in this scene
        np.random.seed(1337 + hash(scene_id) % 1000)
        poses = []
        for _ in range(10):
            pt = pf.get_random_navigable_point()
            poses.append(pt)
            
        print(f"Sampled {len(poses)} poses for scene {scene_id}")
        
        for p_idx, pos in enumerate(poses):
            # Episode dict
            pos_list = [float(pos[0]), float(pos[1]), float(pos[2])]
            ep_dict = {
                "episode_id": f"{scene_id}_pose_{p_idx}",
                "scene_id": scene_id,
                "start_position": pos_list,
                "start_rotation": [0.0, 0.0, 0.0, 1.0],
                "goal_position": [pos_list[0], pos_list[1], pos_list[2] - 5.0],
            }
            obs = emb.reset(ep_dict)
            depth_views = obs.depth_views
            
            # Extract all agent-frame 3D points from 4 views (azimuths 0, 90, 180, 270)
            all_pts = extract_all_valid_agent_points(depth_views, camera_cfg, stride=4, eps=1e-3)
            
            agent = sim.get_agent(0)
            cur_state = agent.get_state()
            pos_sim = cur_state.position
            rot_sim = cur_state.rotation
            
            for th_deg in headings:
                th_rad = np.deg2rad(th_deg)
                for rho_m in distances:
                    # Target vector in agent local frame (forward is -Z, right is +X)
                    local_disp = np.array([rho_m * np.sin(th_rad), 0.0, -rho_m * np.cos(th_rad)], dtype=np.float32)
                    world_disp = quat_rotate_vector(rot_sim, local_disp)
                    target_pos = pos_sim + world_disp
                    
                    # Independent Physical Truth: Simulator NavMesh kinematic step
                    final_pos = pf.try_step(pos_sim, target_pos)
                    dist_moved = float(np.linalg.norm(final_pos - pos_sim))
                    err = float(np.linalg.norm(final_pos - target_pos))
                    
                    # Physically blocked if couldn't traverse within 0.08m tolerance
                    truth_blocked = (dist_moved < (rho_m - 0.08)) or (err > 0.08)
                    
                    # Local frame coordinates for swept rectangle
                    # Project local points along heading (s) and lateral (h)
                    s = all_pts[:, 0] * np.sin(th_rad) - all_pts[:, 2] * np.cos(th_rad)
                    h = all_pts[:, 0] * np.cos(th_rad) + all_pts[:, 2] * np.sin(th_rad)
                    y = all_pts[:, 1] # agent frame Y (height above floor)
                    
                    evaluated_actions.append({
                        "truth_blocked": truth_blocked,
                        "s": s,
                        "h": h,
                        "y": y,
                        "rho_m": rho_m,
                    })
                    
        emb.close()
        print(f"Finished scene {scene_id}, cumulative actions: {len(evaluated_actions)}")
        
    print(f"\nTotal actions evaluated across 30 non-test poses: {len(evaluated_actions)}")
    
    # Sweep grid
    results = []
    total_samples = len(evaluated_actions)
    
    for ra in r_a_grid:
        w = 2 * ra
        for ymin in y_min_grid:
            for nth in n_thresh_grid:
                tp = 0
                tn = 0
                fp = 0
                fn = 0
                
                for act in evaluated_actions:
                    truth_blocked = act["truth_blocked"]
                    s = act["s"]
                    h = act["h"]
                    y = act["y"]
                    rho = act["rho_m"]
                    
                    # Count points in corridor
                    mask = (
                        (y >= ymin) & (y <= y_max) &
                        (s >= 0.0) & (s <= (rho + ra)) &
                        (np.abs(h) <= ra)
                    )
                    cnt = int(np.count_nonzero(mask))
                    pred_blocked = (cnt >= nth)
                    
                    if truth_blocked and pred_blocked:
                        tp += 1
                    elif not truth_blocked and not pred_blocked:
                        tn += 1
                    elif not truth_blocked and pred_blocked:
                        fp += 1
                    elif truth_blocked and not pred_blocked:
                        fn += 1
                        
                acc = (tp + tn) / total_samples
                fbr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Block Rate -> Deadlocks!
                ffr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # False Free Rate -> Collisions
                f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
                
                results.append({
                    "r_a": ra,
                    "w": round(w, 2),
                    "y_min": ymin,
                    "n_thresh": nth,
                    "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                    "accuracy": round(acc, 4),
                    "false_block_rate": round(fbr, 4),
                    "false_free_rate": round(ffr, 4),
                    "f1_score": round(f1, 4),
                })
                
    # Sort by lowest false block rate (deadlock risk)
    results.sort(key=lambda x: (x["false_block_rate"], -x["accuracy"]))
    
    print("\n" + "="*80)
    print("TOP 10 PARAMETER CONFIGURATIONS RANKED BY LOWEST FALSE BLOCK RATE (DEADLOCK AVOIDANCE)")
    print("="*80)
    print(f"{'r_a(m)':<8}{'w(m)':<8}{'y_min':<8}{'N_th':<8}{'Acc':<10}{'FBR(Deadlock)':<16}{'FFR(Bump)':<12}{'F1':<8}")
    for r in results[:10]:
        print(f"{r['r_a']:<8.2f}{r['w']:<8.2f}{r['y_min']:<8.2f}{r['n_thresh']:<8}{r['accuracy']:<10.3f}{r['false_block_rate']:<16.3f}{r['false_free_rate']:<12.3f}{r['f1_score']:<8.3f}")

    # Sort by balanced F1 / Accuracy
    results.sort(key=lambda x: -x["f1_score"])
    print("\n" + "="*80)
    print("TOP 10 PARAMETER CONFIGURATIONS RANKED BY BALANCED F1-SCORE")
    print("="*80)
    print(f"{'r_a(m)':<8}{'w(m)':<8}{'y_min':<8}{'N_th':<8}{'Acc':<10}{'FBR(Deadlock)':<16}{'FFR(Bump)':<12}{'F1':<8}")
    for r in results[:10]:
        print(f"{r['r_a']:<8.2f}{r['w']:<8.2f}{r['y_min']:<8.2f}{r['n_thresh']:<8}{r['accuracy']:<10.3f}{r['false_block_rate']:<16.3f}{r['false_free_rate']:<12.3f}{r['f1_score']:<8.3f}")

    out_file = "audit/interactive_session/independent_calibration_non_test_scenes.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved independent calibration audit to {out_file}")

if __name__ == "__main__":
    run_independent_sweep()
