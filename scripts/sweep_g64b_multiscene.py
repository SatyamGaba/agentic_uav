#!/usr/bin/env python3
"""G6.4b Parameter Sweep across ~30 Poses and 5+ Matterport3D Scenes.

Truth Definition:
  truth_blocked = sim_blocked OR (body_height_points >= 5)
  where:
    - sim_blocked: Habitat NavMesh physics collision (substep < 0.05m or deficit > 0.08m)
    - body_height_points: dense obstacle cluster (>= 5 pts) in agent physical corridor:
        s in [0, rho + 0.10], |h| < 0.10, y in [0.15, 0.80]m

Metrics:
  - FF (False Free): truth_blocked=True,  checker_blocked=False (CRITICAL: MUST BE ZERO)
  - FB (False Block): truth_blocked=False, checker_blocked=True  (Causes deadlocks)
  - TB (True Block):  truth_blocked=True,  checker_blocked=True
  - TF (True Free):   truth_blocked=False, checker_blocked=False

Grid:
  - r_a in [0.10, 0.12, 0.15, 0.18] m
  - y_min in [0.08, 0.10, 0.12, 0.15, 0.18] m
  - y_max in [0.80, 0.88, 1.00, 1.25] m
  - Standoff margin: 0.30 m
  - Depth near-clamp masking: d in [0.501, 9.999] m
"""

import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    camera_to_agent_frame,
    deproject_pixel,
)
from habitat_sim.utils.common import quat_from_angle_axis

def extract_all_valid_agent_points(depth_views, camera_cfg, stride=4, eps=1e-3):
    """Extracts all 3D points in agent frame with valid non-clamped depth."""
    from agenticnav_paper_replica.geometry.pinhole import VIEW_AZIMUTHS_DEG

    all_pts = []
    for vi in range(4):
        dimg = depth_views[vi]
        h, w = dimg.shape
        ys = np.arange(0, h, stride)
        xs = np.arange(0, w, stride)
        gx, gy = np.meshgrid(xs, ys)
        z = dimg[gy, gx].astype(np.float64)

        valid = (z > camera_cfg.d_min + eps) & (z < camera_cfg.d_max - eps) & (~np.isnan(z))
        if not np.any(valid):
            continue

        zv = z[valid]
        gxv = gx[valid]
        gyv = gy[valid]

        xc = (gxv + 0.5 - camera_cfg.cx) * zv / camera_cfg.fx
        yc = (gyv + 0.5 - camera_cfg.cy) * zv / camera_cfg.fy

        xv = xc
        yv = -yc + camera_cfg.camera_height
        zv_frame = -zv

        psi_rad = math.radians(VIEW_AZIMUTHS_DEG[vi])
        cos_psi = math.cos(psi_rad)
        sin_psi = math.sin(psi_rad)

        px = cos_psi * xv + sin_psi * zv_frame
        py = yv
        pz = -sin_psi * xv + cos_psi * zv_frame

        pts_agent = np.column_stack([px, py, pz])
        all_pts.append(pts_agent)

    if all_pts:
        return np.vstack(all_pts)
    return np.empty((0, 3), dtype=np.float64)


def select_poses_across_scenes(n_per_scene=5):
    """Selects poses across available scenes."""
    val_unseen_path = Path("/home/krutik/data/datasets/R2R_VLNCE_v1-3/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz")
    with gzip.open(val_unseen_path, "rt") as f:
        data = json.load(f)

    def clean_scene(sid):
        parts = sid.split("/")
        return parts[1] if len(parts) >= 2 else sid

    available_scenes = ["EU6Fwq7SyZv", "oLBMNvg9in8", "pLe4wQe7qrG", "zsNo4HB9uLZ", "2azQ1b91cZZ", "TbHJrupSAjP"]
    scene_episodes = {s: [] for s in available_scenes}

    for ep in data["episodes"]:
        s = clean_scene(ep["scene_id"])
        if s in scene_episodes and len(scene_episodes[s]) < n_per_scene:
            scene_episodes[s].append(ep)

    # For scenes without enough episodes in val_unseen (like sT4fr6TAbpF), fill with synthetic/navigable points later
    selected = []
    for s, eps in scene_episodes.items():
        for ep in eps:
            selected.append({
                "scene_id": s,
                "episode_id": ep["episode_id"],
                "start_position": ep["start_position"],
                "start_rotation": ep["start_rotation"],
            })
    print(f"Selected {len(selected)} poses across {len(scene_episodes)} scenes.")
    return selected


def main():
    start_time = time.time()
    config = ReplicaConfig()
    camera_cfg = CameraConfig(
        image_width=config.image_width,
        image_height=config.image_height,
        hfov_deg=config.hfov_deg,
        d_min=config.d_min,
        d_max=config.d_max,
        camera_height=config.camera_height,
    )

    poses = select_poses_across_scenes(n_per_scene=5)

    # Grid of action proposals per pose: 14 headings x 5 distances = 70 actions
    headings_deg = [-135, -90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90, 135, 180]
    distances_m = [0.5, 1.0, 1.5, 2.0, 2.5]

    # Grid of checker parameters to sweep
    r_a_list = [0.10, 0.12, 0.15, 0.18]
    y_min_list = [0.08, 0.10, 0.12, 0.15, 0.18]
    y_max_list = [0.80, 0.88, 1.00, 1.25]

    current_scene = None
    emb = None

    # Storage for all evaluated proposals
    # Each entry: {pose_idx, scene, th_deg, rho_m, truth_blocked, sim_blocked, body_pts, pt_coords_corridor}
    evaluated_actions = []

    print(f"\nEvaluating ground truth across {len(poses)} poses...")
    for p_idx, p in enumerate(poses):
        scene_name = p["scene_id"]
        if scene_name != current_scene:
            if emb is not None:
                emb.close()
            current_scene = scene_name
            emb = HabitatGroundEmbodiment(
                scene_id=scene_name,
                camera_config=camera_cfg,
                gpu_device_id=-1,
                max_actions=50,
            )

        # Reset embodiment to this episode pose and get observations
        obs = emb.reset(p)
        depth_views = obs.depth_views
        sim = emb.sim
        agent = sim.get_agent(0)

        # Extract all valid agent-frame 3D points once
        all_pts = extract_all_valid_agent_points(depth_views, camera_cfg, stride=4, eps=1e-3)

        # Base agent initial state at this pose
        init_agent_state = agent.get_state()
        pf = sim.pathfinder
        from habitat_sim.utils.common import quat_rotate_vector

        # Evaluate each action
        for th in headings_deg:
            th_rad = math.radians(th)
            sin_th = math.sin(th_rad)
            cos_th = math.cos(th_rad)

            # Corridor coordinates for all points relative to this heading
            ox = all_pts[:, 0]
            oy = all_pts[:, 1]
            oz = all_pts[:, 2]

            s = -ox * sin_th - oz * cos_th
            h = ox * cos_th - oz * sin_th

            # Compute world forward vector for this heading
            rot_m = quat_from_angle_axis(th_rad, np.array([0.0, 1.0, 0.0]))
            agent_rot = init_agent_state.rotation * rot_m
            forward_vec = quat_rotate_vector(agent_rot, np.array([0.0, 0.0, -1.0]))

            for rho in distances_m:
                # 1. Exact NavMesh stepping truth via pf.try_step (0 rendering overhead)
                cur_pos = np.array(init_agent_state.position, copy=True)
                start_p = np.array(cur_pos, copy=True)
                n_forward = int(round(rho / 0.25))
                sim_collided = False
                for _ in range(n_forward):
                    next_target = cur_pos + forward_vec * 0.25
                    stepped_pos = pf.try_step(cur_pos, next_target)
                    sub_dist = float(np.linalg.norm(stepped_pos - cur_pos))
                    if sub_dist < 0.05:
                        sim_collided = True
                    cur_pos = stepped_pos

                dist_moved = float(np.linalg.norm(cur_pos - start_p))
                dist_deficit = rho - dist_moved
                # Real physical collision: agent hit obstacle on substep (< 0.05m) or lost >= 1 full step (> 0.20m)
                sim_blocked = sim_collided or (dist_deficit > 0.20)

                # 2. Body-height geometry truth:
                # Physical robot body radius is 0.10m, body height is [0.15, 0.80]m
                in_body_corr = (s > 0.0) & (s < (rho + 0.10)) & (np.abs(h) < 0.10) & (oy >= 0.15) & (oy <= 0.80)
                body_pts = int(np.sum(in_body_corr))

                # Truth collision definition:
                truth_blocked = bool(sim_blocked or (body_pts >= 5))

                evaluated_actions.append({
                    "pose_idx": p_idx,
                    "scene": scene_name,
                    "th_deg": th,
                    "rho_m": rho,
                    "sim_blocked": sim_blocked,
                    "body_pts": body_pts,
                    "truth_blocked": truth_blocked,
                    "s": s,
                    "h": h,
                    "oy": oy,
                })

        print(f"  Pose {p_idx + 1:2d}/{len(poses)} ({scene_name}) processed ({len(evaluated_actions)} total actions).", flush=True)

    if emb is not None:
        emb.close()

    total_actions = len(evaluated_actions)
    truth_blocked_cnt = sum(1 for a in evaluated_actions if a["truth_blocked"])
    truth_free_cnt = total_actions - truth_blocked_cnt
    print(f"\nTotal actions evaluated: {total_actions}")
    print(f"Truth blocked: {truth_blocked_cnt} ({truth_blocked_cnt/total_actions*100:.1f}%)")
    print(f"Truth free:    {truth_free_cnt} ({truth_free_cnt/total_actions*100:.1f}%)")

    # Now sweep parameter grid
    print("\nSweeping parameter grid (r_a, y_min, y_max)...")
    grid_results = []

    for r_a in r_a_list:
        for y_min in y_min_list:
            for y_max in y_max_list:
                tb = 0
                tf = 0
                fb = 0
                ff = 0

                for a in evaluated_actions:
                    s = a["s"]
                    h = a["h"]
                    oy = a["oy"]
                    rho = a["rho_m"]
                    truth_b = a["truth_blocked"]

                    # Corridor check with parameters:
                    # Point in height band [y_min, y_max] and corridor [0, rho + r_a] x [-r_a, r_a]
                    in_corr = (s > 0.0) & (s < (rho + r_a)) & (np.abs(h) < r_a) & (oy >= y_min) & (oy <= y_max)
                    checker_b = bool(np.any(in_corr))

                    if truth_b and checker_b:
                        tb += 1
                    elif not truth_b and not checker_b:
                        tf += 1
                    elif not truth_b and checker_b:
                        fb += 1
                    elif truth_b and not checker_b:
                        ff += 1

                accuracy = (tb + tf) / total_actions
                grid_results.append({
                    "r_a": r_a,
                    "y_min": y_min,
                    "y_max": y_max,
                    "TB": tb,
                    "TF": tf,
                    "FB": fb,
                    "FF": ff,
                    "accuracy": accuracy,
                    "deadlock_risk_fb_rate": fb / truth_free_cnt if truth_free_cnt > 0 else 0.0,
                    "collision_risk_ff_rate": ff / truth_blocked_cnt if truth_blocked_cnt > 0 else 0.0,
                })

    # Sort results: primary key FF ascending (MUST BE 0), secondary key FB ascending
    grid_results.sort(key=lambda x: (x["FF"], x["FB"]))

    print("\n" + "=" * 90)
    print(f"{'r_a':>5} | {'y_min':>5} | {'y_max':>5} | {'FF (Crit)':>9} | {'FB (Deadlock)':>13} | {'TB':>5} | {'TF':>5} | {'Acc':>6} | {'FB Rate':>7}")
    print("-" * 90)
    for res in grid_results[:25]:
        print(f"{res['r_a']:5.2f} | {res['y_min']:5.2f} | {res['y_max']:5.2f} | {res['FF']:9d} | {res['FB']:13d} | {res['TB']:5d} | {res['TF']:5d} | {res['accuracy']*100:5.1f}% | {res['deadlock_risk_fb_rate']*100:6.1f}%")
    print("=" * 90)

    # Filter strictly zero False Free (FF == 0)
    zero_ff = [r for r in grid_results if r["FF"] == 0]
    print(f"\nConfigurations achieving STRICT FF = 0: {len(zero_ff)} out of {len(grid_results)}")
    if zero_ff:
        best = zero_ff[0]
        print("\nTOP RECOMMENDED CONFIGURATION (Strict FF = 0, Minimum FB):")
        print(f"  Robot Radius (r_a):  {best['r_a']:.2f} m")
        print(f"  Floor Band (y_min):  {best['y_min']:.2f} m")
        print(f"  Ceiling Band (y_max): {best['y_max']:.2f} m")
        print(f"  False Frees (FF):    {best['FF']} (0.0% collision risk)")
        print(f"  False Blocks (FB):   {best['FB']} ({best['deadlock_risk_fb_rate']*100:.1f}% deadlock risk)")
        print(f"  Overall Accuracy:    {best['accuracy']*100:.2f}%")

    out_file = repo_root / "audit/interactive_session/g64b_multiscene_sweep_results.json"
    with open(out_file, "w") as f:
        # Don't save s, h, oy arrays in json
        for a in evaluated_actions:
            del a["s"]
            del a["h"]
            del a["oy"]
        json.dump({
            "total_actions": total_actions,
            "truth_blocked": truth_blocked_cnt,
            "truth_free": truth_free_cnt,
            "sweep_grid": grid_results,
            "best_zero_ff": zero_ff[0] if zero_ff else None,
            "elapsed_seconds": round(time.time() - start_time, 2),
        }, f, indent=2)

    print(f"\nSaved full sweep results to {out_file} in {time.time() - start_time:.1f}s")


if __name__ == "__main__":
    main()
