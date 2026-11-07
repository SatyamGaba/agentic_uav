#!/usr/bin/env python3
"""Empirical Floor Calibration and Masked Corridor Sweep for Episode 0764.

Addresses:
1. Floor Plane Fitting: Measures floor datum height and residual distribution vs range [0.5, 5.0] m.
2. 5 Benchmark Targets Re-evaluation: Evaluates the 5 rejected targets at their true rho_q with near-clamp masking.
3. 96-Proposal Sweep Re-run: Categorizes obstacle rejections by cause:
   - Near-plane clamped depth (d <= d_min + eps)
   - Floor grazing noise (points near floor plane)
   - Real body-height obstacles (y in [0.15, 1.25]m with valid depth)
4. Fast actuation using direct quaternion rotation to avoid slow CPU rasterization.
"""

import copy
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure repo root is on python path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    VIEW_AZIMUTHS_DEG,
    camera_to_agent_frame,
    deproject_pixel,
    extract_view_obstacle_and_free_points,
    normalize_to_pixel,
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker


def main():
    print("=" * 80)
    print("STEP 2: FLOOR PLANE CALIBRATION & ELEVATION RESIDUALS VS. RANGE")
    print("=" * 80)

    config = ReplicaConfig()
    camera_cfg = CameraConfig(
        image_width=config.image_width,
        image_height=config.image_height,
        hfov_deg=config.hfov_deg,
        d_min=config.d_min,
        d_max=config.d_max,
        camera_height=config.camera_height,
    )

    dataset_file = repo_root / "agenticnav_paper_replica/data/r2rce_val_unseen_17.json"
    with open(dataset_file) as f:
        target_ep = next(e for e in json.load(f)["episodes"] if "0764" in str(e["episode_id"]))

    emb = HabitatGroundEmbodiment(
        scene_id=target_ep["scene_id"],
        camera_config=camera_cfg,
        gpu_device_id=-1,
        max_actions=50,
    )
    obs = emb.reset(target_ep)
    sim = emb.sim
    agent = sim.get_agent(0)
    initial_state = agent.get_state()

    # 1. Floor calibration: extract points in lower half of image (v > 0.55) across all views
    # An un-tilted camera at 1.25m height sees floor only when v > 0.50
    floor_pts_list = []
    eps = 1e-3
    for vi in range(4):
        dimg = obs.depth_views[vi]
        h, w = dimg.shape
        # Sample lower rows: y from 260 to 475 (stride 2)
        ys = np.arange(260, h - 5, 2)
        xs = np.arange(10, w - 10, 2)
        grid_x, grid_y = np.meshgrid(xs, ys)
        z_vals = dimg[grid_y, grid_x].astype(np.float64)

        # Valid in-range depth (not clamped)
        valid = (z_vals > camera_cfg.d_min + eps) & (z_vals < camera_cfg.d_max - eps) & (~np.isnan(z_vals))
        if not np.any(valid):
            continue

        z_v = z_vals[valid]
        gx = grid_x[valid]
        gy = grid_y[valid]

        x_c = (gx + 0.5 - camera_cfg.cx) * z_v / camera_cfg.fx
        y_c = (gy + 0.5 - camera_cfg.cy) * z_v / camera_cfg.fy

        x_v = x_c
        y_v = -y_c + camera_cfg.camera_height
        z_v_opt = -z_v

        psi_rad = math.radians(VIEW_AZIMUTHS_DEG[vi])
        cos_psi = math.cos(psi_rad)
        sin_psi = math.sin(psi_rad)

        px = cos_psi * x_v + sin_psi * z_v_opt
        py = y_v
        pz = -sin_psi * x_v + cos_psi * z_v_opt

        # Candidates for floor points: py < 0.25m
        cand_floor = py < 0.25
        if np.any(cand_floor):
            floor_pts_list.append(np.column_stack([px[cand_floor], py[cand_floor], pz[cand_floor]]))

    all_floor_pts = np.vstack(floor_pts_list)
    print(f"Extracted {len(all_floor_pts)} floor-candidate points across panoramic views.")

    # Measure floor height py statistics
    floor_y = all_floor_pts[:, 1]
    floor_dists = np.sqrt(all_floor_pts[:, 0] ** 2 + all_floor_pts[:, 2] ** 2)

    print(f"Floor Elevation (y above contact datum):")
    print(f"  Mean:   {np.mean(floor_y):.4f} m")
    print(f"  Std:    {np.std(floor_y):.4f} m")
    print(f"  Median: {np.median(floor_y):.4f} m")
    print(f"  Range:  [{np.min(floor_y):.4f} m, {np.max(floor_y):.4f} m]")

    # Measure floor height vs range bins
    range_bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 4.5)]
    print("\nFloor Elevation by Range Bin:")
    floor_bin_stats = {}
    for r_lo, r_hi in range_bins:
        in_bin = (floor_dists >= r_lo) & (floor_dists < r_hi)
        if np.sum(in_bin) > 0:
            y_bin = floor_y[in_bin]
            print(f"  Range [{r_lo:.1f}m, {r_hi:.1f}m]: N={np.sum(in_bin)}, Mean={np.mean(y_bin):.4f}m, Std={np.std(y_bin):.4f}m, Max={np.max(y_bin):.4f}m")
            floor_bin_stats[f"{r_lo}-{r_hi}"] = {
                "count": int(np.sum(in_bin)),
                "mean": float(np.mean(y_bin)),
                "std": float(np.std(y_bin)),
                "max": float(np.max(y_bin)),
            }

    print("\n" + "=" * 80)
    print("STEP 1: RE-EVALUATE 5 BENCHMARK TARGETS AT TRUE rho_q WITH CLAMP MASKED")
    print("=" * 80)

    # Extract clean obstacle pointcloud with near-plane clamp MASKED
    clean_obs_list = []
    raw_obs_list = []
    for vi in range(4):
        dimg = obs.depth_views[vi]
        # 1. Clean (masking <= 0.501m)
        pts_clean, _ = extract_view_obstacle_and_free_points(
            dimg, view_idx=vi, config=camera_cfg, stride=4, y_min=config.chassis_y_min, y_max=config.chassis_y_max, eps=1e-3
        )
        if len(pts_clean) > 0:
            clean_obs_list.append(pts_clean)

        # 2. Raw unmasked (old behaviour)
        pts_raw, _ = extract_view_obstacle_and_free_points(
            dimg, view_idx=vi, config=camera_cfg, stride=4, y_min=config.chassis_y_min, y_max=config.chassis_y_max, eps=-1e-3
        )
        if len(pts_raw) > 0:
            raw_obs_list.append(pts_raw)

    clean_obs_3d = np.vstack(clean_obs_list) if clean_obs_list else np.empty((0, 3))
    clean_obs_2d = clean_obs_3d[:, [0, 2]] if len(clean_obs_3d) > 0 else np.empty((0, 2))
    raw_obs_3d = np.vstack(raw_obs_list) if raw_obs_list else np.empty((0, 3))
    raw_obs_2d = raw_obs_3d[:, [0, 2]] if len(raw_obs_3d) > 0 else np.empty((0, 2))

    print(f"Total raw obstacles extracted:   {len(raw_obs_3d)}")
    print(f"Total clean obstacles (no clamp): {len(clean_obs_3d)} (Removed {len(raw_obs_3d) - len(clean_obs_3d)} clamped artifacts)")

    checker_r18 = GroundCorridorChecker(robot_radius=0.18, base_margin=0.30, rho_max=2.5)
    checker_r10 = GroundCorridorChecker(robot_radius=0.10, base_margin=0.30, rho_max=2.5)

    attempted_targets = [
        {"turn": 2, "view": 0, "u": 0.50, "v": 0.50, "label": "Turn 2 (v=0, u=0.50, v=0.50)"},
        {"turn": 3, "view": 0, "u": 0.65, "v": 0.50, "label": "Turn 3 (v=0, u=0.65, v=0.50)"},
        {"turn": 4, "view": 0, "u": 0.75, "v": 0.50, "label": "Turn 4 (v=0, u=0.75, v=0.50)"},
        {"turn": 5, "view": 0, "u": 0.25, "v": 0.50, "label": "Turn 5 (v=0, u=0.25, v=0.50)"},
        {"turn": 6, "view": 3, "u": 0.50, "v": 0.50, "label": "Turn 6 (v=3, u=0.50, v=0.50)"},
    ]

    target_retest_results = []
    for t in attempted_targets:
        vi = t["view"]
        u, v = t["u"], t["v"]
        dimg = obs.depth_views[vi]
        x0, y0 = normalize_to_pixel(u, v, camera_cfg)
        raw_depth = float(dimg[y0, x0])
        p_c = deproject_pixel(x0, y0, raw_depth, camera_cfg, is_normalized=False)
        p_agent = camera_to_agent_frame(p_c, vi, camera_cfg)
        px, py, pz = float(p_agent[0]), float(p_agent[1]), float(p_agent[2])

        # Compute true kinematics
        d_raw, th_deg, true_rho_q, true_th_q = checker_r18.compute_kinematics(px, pz)

        # Evaluate with raw vs clean obstacles
        res_raw_18 = checker_r18.evaluate(p_x=px, p_z=pz, obstacles_2d=raw_obs_2d)
        res_clean_18 = checker_r18.evaluate(p_x=px, p_z=pz, obstacles_2d=clean_obs_2d)
        res_clean_10 = checker_r10.evaluate(p_x=px, p_z=pz, obstacles_2d=clean_obs_2d)

        # Decompose collisions by height band
        theta_rad = math.radians(true_th_q)
        sin_th = math.sin(theta_rad)
        cos_th = math.cos(theta_rad)
        ox = clean_obs_3d[:, 0]
        oy = clean_obs_3d[:, 1]
        oz = clean_obs_3d[:, 2]

        s = -ox * sin_th - oz * cos_th
        h = ox * cos_th - oz * sin_th
        in_corr = (s > 0.0) & (s < (true_rho_q + 0.18)) & (np.abs(h) < 0.18)

        floor_leak_pts = np.sum(in_corr & (oy < 0.15))
        mid_chassis_pts = np.sum(in_corr & (oy >= 0.15) & (oy < 0.80))
        high_overhang_pts = np.sum(in_corr & (oy >= 0.80))

        item = {
            "label": t["label"],
            "depth_m": raw_depth,
            "true_rho_q": true_rho_q,
            "true_th_q": true_th_q,
            "raw_collisions_w36": res_raw_18.collision_point_count,
            "clean_collisions_w36": res_clean_18.collision_point_count,
            "clean_status_w36": res_clean_18.status,
            "clean_collisions_w20": res_clean_10.collision_point_count,
            "clean_status_w20": res_clean_10.status,
            "collision_breakdown": {
                "floor_leak_y_lt_015": int(floor_leak_pts),
                "mid_chassis_015_080": int(mid_chassis_pts),
                "high_overhang_y_ge_080": int(high_overhang_pts),
            },
        }
        target_retest_results.append(item)

        print(f"\n{t['label']}: True rho_q={true_rho_q:.2f}m, True theta_q={true_th_q:.1f}°")
        print(f"  Raw Obstacles (w=0.36m):   status={res_raw_18.status}, coll={res_raw_18.collision_point_count}")
        print(f"  Clean Obstacles (w=0.36m): status={res_clean_18.status}, coll={res_clean_18.collision_point_count}")
        print(f"  Clean Obstacles (w=0.20m): status={res_clean_10.status}, coll={res_clean_10.collision_point_count}")
        print(f"  Collision Breakdown in Corridor: floor_leak(<0.15m)={floor_leak_pts}, mid_chassis(0.15-0.80m)={mid_chassis_pts}, high_overhang(>=0.80m)={high_overhang_pts}")

    print("\n" + "=" * 80)
    print("STEP 3: FAST 96-PROPOSAL SWEEP WITH CLAMP MASKED & DECOMPOSED CAUSES")
    print("=" * 80)

    # 24 Headings x 4 Distances
    headings_deg = [k * 15 for k in range(-12, 12)]
    distances_m = [0.5, 1.0, 1.5, 2.0]

    sweep_records = []
    for th in headings_deg:
        th_rad = math.radians(th)
        sin_th = math.sin(th_rad)
        cos_th = math.cos(th_rad)

        for rho in distances_m:
            # 1. Clean Checker evaluation at w=0.36m and w=0.20m
            coll_18, count_18 = checker_r18.test_corridor_collision(
                theta_deg=float(th), rho_q=rho, obstacles_2d=clean_obs_2d
            )
            coll_10, count_10 = checker_r10.test_corridor_collision(
                theta_deg=float(th), rho_q=rho, obstacles_2d=clean_obs_2d
            )

            # Categorize causes of rejection if blocked at w=0.36m
            ox = clean_obs_3d[:, 0]
            oy = clean_obs_3d[:, 1]
            oz = clean_obs_3d[:, 2]
            s = -ox * sin_th - oz * cos_th
            h = ox * cos_th - oz * sin_th
            in_corr = (s > 0.0) & (s < (rho + 0.18)) & (np.abs(h) < 0.18)

            n_floor = int(np.sum(in_corr & (oy < 0.15)))
            n_mid = int(np.sum(in_corr & (oy >= 0.15) & (oy < 0.80)))
            n_high = int(np.sum(in_corr & (oy >= 0.80)))

            if count_18 == 0:
                cause = "none"
            elif n_floor > 0 and n_mid == 0 and n_high == 0:
                cause = "floor_leak_only"
            elif n_high > 0 and n_mid == 0 and n_floor == 0:
                cause = "high_overhang_only"
            elif n_mid > 0:
                cause = "body_height_obstacle"
            else:
                cause = "mixed"

            # Fast Simulator check using direct set_state quaternion rotation
            agent.set_state(initial_state)
            # Direct rotation around Y axis
            half_yaw = th_rad * 0.5
            rot_quat = [0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)]
            cur_state = agent.get_state()
            # Habitat quaternion format: [x, y, z, w] or habitat_sim.utils.common.quat_from_angle_axis
            from habitat_sim.utils.common import quat_from_angle_axis, quat_to_magnum
            rot_m = quat_from_angle_axis(th_rad, np.array([0.0, 1.0, 0.0]))
            cur_state.rotation = rot_m
            agent.set_state(cur_state)

            pre_pos = np.array(agent.get_state().position, copy=True)
            n_forward = int(round(rho / 0.25))
            sim_collided = False
            for _ in range(n_forward):
                p_sub_pre = np.array(agent.get_state().position, copy=True)
                sim.step("move_forward")
                p_sub_post = np.array(agent.get_state().position, copy=True)
                if np.linalg.norm(p_sub_post - p_sub_pre) < 0.05:
                    sim_collided = True

            post_pos = np.array(agent.get_state().position, copy=True)
            dist_moved = float(np.linalg.norm(post_pos - pre_pos))
            dist_deficit = rho - dist_moved
            sim_blocked = sim_collided or (dist_deficit > 0.08)

            sweep_records.append({
                "heading_deg": th,
                "target_dist_m": rho,
                "clean_w36_blocked": coll_18,
                "clean_w36_coll_count": count_18,
                "clean_w20_blocked": coll_10,
                "clean_w20_coll_count": count_10,
                "rejection_cause": cause,
                "floor_pts": n_floor,
                "mid_pts": n_mid,
                "high_pts": n_high,
                "sim_dist_moved": round(dist_moved, 3),
                "sim_collided": sim_collided,
                "sim_blocked": sim_blocked,
            })

    # Summary by distance rho
    print("\nSweep Re-run Summary by Target Distance rho (with Near-Clamp Masked):")
    for rho in distances_m:
        recs = [r for r in sweep_records if abs(r["target_dist_m"] - rho) < 1e-4]
        n_p = len(recs)
        sim_free = sum(1 for r in recs if not r["sim_blocked"])
        w36_blocked = sum(1 for r in recs if r["clean_w36_blocked"])
        w20_blocked = sum(1 for r in recs if r["clean_w20_blocked"])
        fb_36 = sum(1 for r in recs if r["clean_w36_blocked"] and not r["sim_blocked"])
        fb_20 = sum(1 for r in recs if r["clean_w20_blocked"] and not r["sim_blocked"])
        ff_36 = sum(1 for r in recs if not r["clean_w36_blocked"] and r["sim_blocked"])
        ff_20 = sum(1 for r in recs if not r["clean_w20_blocked"] and r["sim_blocked"])

        print(f"  rho = {rho:.1f}m (N={n_p}):")
        print(f"    Sim Free: {sim_free}/{n_p} | w=0.36m Blocked: {w36_blocked} (FB: {fb_36}, FF: {ff_36}) | w=0.20m Blocked: {w20_blocked} (FB: {fb_20}, FF: {ff_20})")

    # Aggregate causes of false blocks
    all_fb_recs = [r for r in sweep_records if r["clean_w36_blocked"] and not r["sim_blocked"]]
    cause_counts = {}
    for r in all_fb_recs:
        c = r["rejection_cause"]
        cause_counts[c] = cause_counts.get(c, 0) + 1
    print(f"\nDecomposition of False Blocks (N={len(all_fb_recs)}):")
    for c, cnt in cause_counts.items():
        print(f"  {c}: {cnt} ({cnt/len(all_fb_recs)*100:.1f}%)")

    # Save results to JSON
    out_file = repo_root / "audit/interactive_session/floor_calibration_and_resweep.json"
    with open(out_file, "w") as f:
        json.dump({
            "floor_calibration": {
                "mean_y": float(np.mean(floor_y)),
                "std_y": float(np.std(floor_y)),
                "range_bins": floor_bin_stats,
            },
            "target_retests": target_retest_results,
            "resweep_records": sweep_records,
        }, f, indent=2)
    print(f"\nSaved calibration and re-sweep results to {out_file}")

    emb.close()


if __name__ == "__main__":
    main()

