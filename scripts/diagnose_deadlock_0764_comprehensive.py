#!/usr/bin/env python3
"""Comprehensive empirical diagnostic for Episode 0764 deadlock.

Investigates:
1. Runtime Parameter Dump & Configuration Audit.
2. 24-Heading empirical sweep comparing GroundCorridorChecker vs. Habitat-Sim physical actuation.
3. Analysis of the 5 failed targets from the official benchmark trace.
4. Visual back-projection overlay onto view_0 and 2D X-Z corridor scatter.
"""

import copy
import json
import math
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

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
    print("STEP 1: RUNTIME PARAMETER DUMP & CONFIGURATION AUDIT")
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
    checker_default = GroundCorridorChecker(
        robot_radius=config.robot_radius,
        base_margin=config.base_margin,
        rho_max=config.rho_max,
        turn_step_deg=config.turn_step_deg,
        forward_step_m=config.forward_step_m,
    )
    checker_thin = GroundCorridorChecker(
        robot_radius=0.10,  # Habitat-sim default agent radius
        base_margin=config.base_margin,
        rho_max=config.rho_max,
        turn_step_deg=config.turn_step_deg,
        forward_step_m=config.forward_step_m,
    )

    print(f"GroundCorridorChecker (default code):")
    print(f"  self.r_a              = {checker_default.r_a:.3f} m (robot radius)")
    print(f"  self.base_margin      = {checker_default.base_margin:.3f} m (standoff margin)")
    print(f"  Total corridor width  = {2 * checker_default.r_a:.3f} m (lateral check: |h| < r_a)")
    print(f"  Margin folded in?     = NO (margin is subtracted from longitudinal rho = D - m)")
    print(f"  Spec ledger assumption= 0.48m half-width (0.96m total) -> DISCREPANCY: code uses 0.36m!")
    print(f"Camera & Chassis Config:")
    print(f"  config.camera_height  = {config.camera_height:.3f} m")
    print(f"  config.d_max          = {config.d_max:.3f} m")
    print(f"  config.d_min          = {config.d_min:.3f} m")
    print(f"  config.chassis_y_min  = {config.chassis_y_min:.3f} m")
    print(f"  config.chassis_y_max  = {config.chassis_y_max:.3f} m")
    print(f"Habitat-Sim standard agent configuration defaults:")
    print(f"  habitat_sim.agent.AgentConfiguration.radius = 0.10 m (total diameter = 0.20 m)")
    print(f"  habitat_sim.agent.AgentConfiguration.height = 1.50 m")
    print(f"  Camera mount position = [0.0, 1.25, 0.0] (matching config.camera_height)")

    # Load episode 0764
    dataset_file = repo_root / "agenticnav_paper_replica/data/r2rce_val_unseen_17.json"
    with open(dataset_file) as f:
        data = json.load(f)
    episodes = data["episodes"]
    target_ep = next(e for e in episodes if "0764" in str(e["episode_id"]))

    print("\nTarget Episode Details:")
    print(f"  Episode ID: {target_ep['episode_id']}")
    print(f"  Scene ID:   {target_ep['scene_id']}")
    print(f"  Start Pos:  {target_ep['start_position']}")
    print(f"  Start Rot:  {target_ep['start_rotation']}")
    print(f"  Instruction:{target_ep.get('instruction')}")

    # Launch Habitat embodiment
    print("\nInitializing Habitat-Sim embodiment on scene EU6Fwq7SyZv...")
    embodiment = HabitatGroundEmbodiment(
        scene_id=target_ep["scene_id"],
        camera_config=camera_cfg,
        gpu_device_id=-1,
        max_actions=50,
    )
    obs = embodiment.reset(target_ep)
    sim = embodiment.sim
    agent = sim.get_agent(0)
    initial_state = agent.get_state()
    initial_pos = np.array(initial_state.position)
    initial_rot = copy.deepcopy(initial_state.rotation)

    print(f"Simulator reset successful. Agent initial pos: {initial_pos}")

    # Extract point clouds from depth views
    obs_agent_list = []
    view_pts_dict = {}
    for vi, dimg in obs.depth_views.items():
        pts_obs, pts_free = extract_view_obstacle_and_free_points(
            dimg, view_idx=vi, config=camera_cfg, stride=4, y_min=config.chassis_y_min, y_max=config.chassis_y_max
        )
        view_pts_dict[vi] = (pts_obs, pts_free)
        if len(pts_obs) > 0:
            obs_agent_list.append(pts_obs)

    all_obs_3d = np.vstack(obs_agent_list) if obs_agent_list else np.empty((0, 3))
    obs_2d = all_obs_3d[:, [0, 2]] if len(all_obs_3d) > 0 else np.empty((0, 2))
    print(f"Total extracted 3D obstacle points across 4 views: {len(all_obs_3d)}")

    # Check obstacle distribution by view
    for vi in range(4):
        pts = view_pts_dict[vi][0]
        if len(pts) > 0:
            dists = np.linalg.norm(pts[:, [0, 2]], axis=1)
            close = pts[dists < 1.5]
            print(f"  View {vi} ({VIEW_AZIMUTHS_DEG[vi]}°): {len(pts)} total obstacles, {len(close)} within 1.5m, y range [{np.min(pts[:, 1]):.3f}, {np.max(pts[:, 1]):.3f}]")

    print("\n" + "=" * 80)
    print("STEP 2: ANALYSIS OF THE 5 ATTEMPTED BENCHMARK TARGETS")
    print("=" * 80)
    attempted_targets = [
        {"turn": 2, "view": 0, "u": 0.50, "v": 0.50, "label": "Turn 2 (v=0, u=0.5, v=0.5)"},
        {"turn": 3, "view": 0, "u": 0.65, "v": 0.50, "label": "Turn 3 (v=0, u=0.65, v=0.5)"},
        {"turn": 4, "view": 0, "u": 0.75, "v": 0.50, "label": "Turn 4 (v=0, u=0.75, v=0.5)"},
        {"turn": 5, "view": 0, "u": 0.25, "v": 0.50, "label": "Turn 5 (v=0, u=0.25, v=0.5)"},
        {"turn": 6, "view": 3, "u": 0.50, "v": 0.50, "label": "Turn 6 (v=3, u=0.5, v=0.5)"},
    ]

    target_eval_results = []
    for t in attempted_targets:
        vi = t["view"]
        u = t["u"]
        v = t["v"]
        dimg = obs.depth_views[vi]
        x0, y0 = normalize_to_pixel(u, v, camera_cfg)
        raw_depth = float(dimg[y0, x0])
        p_c = deproject_pixel(x0, y0, raw_depth, camera_cfg, is_normalized=False)
        p_agent = camera_to_agent_frame(p_c, vi, camera_cfg)
        px, py, pz = float(p_agent[0]), float(p_agent[1]), float(p_agent[2])
        raw_dist = math.sqrt(px * px + pz * pz)

        # Theoretical floor horizon comparison
        # At v=0.5, y_c = 0 -> y_agent = camera_height = 1.25m!
        # An actual floor point would have y_c = 1.25m -> y_agent = 0.0m -> v_floor = 0.5 + 0.833/d
        v_floor_theor = 0.5 + (camera_cfg.camera_height * camera_cfg.fy) / (camera_cfg.image_height * max(0.5, raw_depth))

        res_default = checker_default.evaluate(p_x=px, p_z=pz, obstacles_2d=obs_2d)
        res_thin = checker_thin.evaluate(p_x=px, p_z=pz, obstacles_2d=obs_2d)

        # Identify which points collide
        theta_rad = math.atan2(-px, -pz)
        theta_deg = math.degrees(theta_rad)
        sin_th = math.sin(theta_rad)
        cos_th = math.cos(theta_rad)
        ox = obs_2d[:, 0]
        oz = obs_2d[:, 1]
        s = -ox * sin_th - oz * cos_th
        h = ox * cos_th - oz * sin_th
        coll_mask = (s > 0.0) & (s < (res_default.quantized_rho_m + checker_default.r_a)) & (np.abs(h) < checker_default.r_a)
        coll_points_3d = all_obs_3d[coll_mask]

        t_res = {
            "label": t["label"],
            "view": vi,
            "u": u,
            "v": v,
            "depth_m": raw_depth,
            "agent_xyz": [round(px, 3), round(py, 3), round(pz, 3)],
            "target_height_m": round(py, 3),
            "v_floor_theoretical": round(v_floor_theor, 3),
            "checker_r18": {
                "status": res_default.status,
                "cause": res_default.cause,
                "rho_q": res_default.quantized_rho_m,
                "theta_q": res_default.quantized_theta_deg,
                "collisions": res_default.collision_point_count,
            },
            "checker_r10": {
                "status": res_thin.status,
                "cause": res_thin.cause,
                "rho_q": res_thin.quantized_rho_m,
                "theta_q": res_thin.quantized_theta_deg,
                "collisions": res_thin.collision_point_count,
            },
            "colliding_points_count": int(np.sum(coll_mask)),
            "colliding_y_min": float(np.min(coll_points_3d[:, 1])) if len(coll_points_3d) > 0 else None,
            "colliding_y_max": float(np.max(coll_points_3d[:, 1])) if len(coll_points_3d) > 0 else None,
            "colliding_s_min": float(np.min(s[coll_mask])) if len(coll_points_3d) > 0 else None,
            "colliding_s_max": float(np.max(s[coll_mask])) if len(coll_points_3d) > 0 else None,
        }
        target_eval_results.append(t_res)

        print(f"\n{t['label']}:")
        print(f"  Pixel: (u={u}, v={v}), Depth: {raw_depth:.2f} m")
        print(f"  Agent 3D Coord: [{px:.2f}, {py:.2f}, {pz:.2f}] (Height above floor: {py:.2f} m!)")
        print(f"  (Theoretical floor pixel for depth {raw_depth:.2f}m would be v={v_floor_theor:.3f}, NOT v=0.500!)")
        print(f"  Checker (r=0.18m, w=0.36m): status={res_default.status}, cause={res_default.cause}, collisions={res_default.collision_point_count}")
        print(f"  Checker (r=0.10m, w=0.20m): status={res_thin.status}, cause={res_thin.cause}, collisions={res_thin.collision_point_count}")
        if len(coll_points_3d) > 0:
            print(f"  Colliding points inside corridor: N={len(coll_points_3d)}")
            print(f"  Corridor longitudinal distance s: [{np.min(s[coll_mask]):.2f}m, {np.max(s[coll_mask]):.2f}m]")
            print(f"  Obstacle height y: [{np.min(coll_points_3d[:, 1]):.2f}m, {np.max(coll_points_3d[:, 1]):.2f}m]")

    print("\n" + "=" * 80)
    print("STEP 3: 24-HEADING EMPIRICAL SWEEP (G6.4b SINGLE POSE)")
    print("=" * 80)
    print("Evaluating 24 radial headings x 4 step distances against Habitat-Sim physics...")

    # Headings: 24 increments of 15 degrees (-180 to 165)
    headings_deg = [k * 15 for k in range(-12, 12)]  # [-180, -165, ..., 150, 165]
    distances_m = [0.5, 1.0, 1.5, 2.0]

    sweep_records = []

    for th in headings_deg:
        for rho in distances_m:
            # 1. Checker evaluation at r_a = 0.18m
            coll_18, count_18 = checker_default.test_corridor_collision(
                theta_deg=float(th), rho_q=rho, obstacles_2d=obs_2d
            )
            # 2. Checker evaluation at r_a = 0.10m
            coll_10, count_10 = checker_thin.test_corridor_collision(
                theta_deg=float(th), rho_q=rho, obstacles_2d=obs_2d
            )

            # 3. Habitat-Sim physical execution
            # Reset agent state
            agent.set_state(initial_state)

            # Rotate to heading th: in habitat, negative yaw = right, positive yaw = left
            # Turn step is 15 degrees
            n_turns = int(round(th / 15.0))
            if n_turns > 0:
                for _ in range(n_turns):
                    sim.step("turn_left")
            elif n_turns < 0:
                for _ in range(abs(n_turns)):
                    sim.step("turn_right")

            pre_move_pos = np.array(agent.get_state().position)

            # Step forward n_steps
            n_forward = int(round(rho / 0.25))
            sim_collided = False
            for step_i in range(n_forward):
                p_sub_pre = np.array(agent.get_state().position, copy=True)
                sim.step("move_forward")
                p_sub_post = np.array(agent.get_state().position, copy=True)
                if np.linalg.norm(p_sub_post - p_sub_pre) < 0.05:
                    sim_collided = True

            post_move_pos = np.array(agent.get_state().position)
            # In Habitat coordinates: x is right, y is up, z is depth/forward
            actual_dist_moved = float(np.linalg.norm(post_move_pos - pre_move_pos))
            dist_deficit = rho - actual_dist_moved

            # Did simulator successfully complete the motion?
            # Free if deficit < 0.05m and not sim_collided
            sim_blocked = sim_collided or (dist_deficit > 0.08)

            rec = {
                "heading_deg": th,
                "target_dist_m": rho,
                "checker_w36_blocked": coll_18,
                "checker_w36_coll_count": count_18,
                "checker_w20_blocked": coll_10,
                "checker_w20_coll_count": count_10,
                "sim_dist_moved": round(actual_dist_moved, 3),
                "sim_collided": sim_collided,
                "sim_blocked": sim_blocked,
            }
            sweep_records.append(rec)

    # Print summary table of headings for rho = 1.0m and rho = 1.5m
    print("\n24-Heading Results for rho = 1.0 m:")
    print(f"{'Heading':>8} | {'Checker w=0.36m':>15} | {'Checker w=0.20m':>15} | {'Sim Moved':>10} | {'Sim Collided':>12} | {'Verdict':>12}")
    print("-" * 85)
    for rec in sweep_records:
        if abs(rec["target_dist_m"] - 1.0) < 1e-4:
            c18 = f"BLOCKED ({rec['checker_w36_coll_count']})" if rec["checker_w36_blocked"] else "FREE"
            c10 = f"BLOCKED ({rec['checker_w20_coll_count']})" if rec["checker_w20_blocked"] else "FREE"
            v = "AGREE" if (rec["checker_w36_blocked"] == rec["sim_blocked"]) else ("FALSE_BLOCK" if rec["checker_w36_blocked"] else "FALSE_FREE")
            print(f"{rec['heading_deg']:>7}° | {c18:>15} | {c10:>15} | {rec['sim_dist_moved']:>9.2f}m | {str(rec['sim_collided']):>12} | {v:>12}")

    print("\n24-Heading Results for rho = 1.5 m:")
    print(f"{'Heading':>8} | {'Checker w=0.36m':>15} | {'Checker w=0.20m':>15} | {'Sim Moved':>10} | {'Sim Collided':>12} | {'Verdict':>12}")
    print("-" * 85)
    for rec in sweep_records:
        if abs(rec["target_dist_m"] - 1.5) < 1e-4:
            c18 = f"BLOCKED ({rec['checker_w36_coll_count']})" if rec["checker_w36_blocked"] else "FREE"
            c10 = f"BLOCKED ({rec['checker_w20_coll_count']})" if rec["checker_w20_blocked"] else "FREE"
            v = "AGREE" if (rec["checker_w36_blocked"] == rec["sim_blocked"]) else ("FALSE_BLOCK" if rec["checker_w36_blocked"] else "FALSE_FREE")
            print(f"{rec['heading_deg']:>7}° | {c18:>15} | {c10:>15} | {rec['sim_dist_moved']:>9.2f}m | {str(rec['sim_collided']):>12} | {v:>12}")

    # Aggregate metrics across all 96 proposals
    n_total = len(sweep_records)
    c18_blocked = sum(1 for r in sweep_records if r["checker_w36_blocked"])
    c10_blocked = sum(1 for r in sweep_records if r["checker_w20_blocked"])
    sim_blocked_count = sum(1 for r in sweep_records if r["sim_blocked"])
    false_blocks_18 = sum(1 for r in sweep_records if r["checker_w36_blocked"] and not r["sim_blocked"])
    false_blocks_10 = sum(1 for r in sweep_records if r["checker_w20_blocked"] and not r["sim_blocked"])
    false_free_18 = sum(1 for r in sweep_records if not r["checker_w36_blocked"] and r["sim_blocked"])
    false_free_10 = sum(1 for r in sweep_records if not r["checker_w20_blocked"] and r["sim_blocked"])

    print("\nAggregate Sweep Statistics (24 headings x 4 distances = 96 proposals):")
    print(f"  Simulator Actual Blocked / Collided: {sim_blocked_count} / {n_total} ({sim_blocked_count/n_total*100:.1f}%)")
    print(f"  Simulator Freely Accessible Paths:   {n_total - sim_blocked_count} / {n_total} ({(n_total-sim_blocked_count)/n_total*100:.1f}%)")
    print(f"  Checker (w=0.36m, r=0.18m) Blocked:  {c18_blocked} / {n_total} ({c18_blocked/n_total*100:.1f}%)")
    print(f"    - False Block Rate:                {false_blocks_18} / {n_total - sim_blocked_count} ({false_blocks_18/max(1, n_total - sim_blocked_count)*100:.1f}%)")
    print(f"    - False Free Rate:                 {false_free_18} / {max(1, sim_blocked_count)} ({false_free_18/max(1, sim_blocked_count)*100:.1f}%)")
    print(f"  Checker (w=0.20m, r=0.10m) Blocked:  {c10_blocked} / {n_total} ({c10_blocked/n_total*100:.1f}%)")
    print(f"    - False Block Rate:                {false_blocks_10} / {n_total - sim_blocked_count} ({false_blocks_10/max(1, n_total - sim_blocked_count)*100:.1f}%)")
    print(f"    - False Free Rate:                 {false_free_10} / {max(1, sim_blocked_count)} ({false_free_10/max(1, sim_blocked_count)*100:.1f}%)")

    print("\n" + "=" * 80)
    print("STEP 4: GENERATE VISUAL BACK-PROJECTION & 2D SCATTER PLOTS")
    print("=" * 80)
    out_dir = repo_root / "audit/interactive_session"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 4a. Back-projection onto view_0.png
    # Load view 0 RGB image
    rgb_v0 = obs.rgb_views[0]  # (480, 640, 3 or 4)
    if rgb_v0.shape[2] == 4:
        rgb_v0 = rgb_v0[:, :, :3]
    h_img, w_img, _ = rgb_v0.shape

    # Pinhole reprojection function from agent coordinates [p_x, p_y, p_z] to view_0 pixel coordinates
    # For view 0 (psi = 0 deg):
    # p_x = x_c, p_y = -y_c + h_cam, p_z = -z_c
    # x_c = p_x, y_c = h_cam - p_y, z_c = -p_z
    # u_px = x_c * fx / z_c + cx - 0.5
    # v_px = y_c * fy / z_c + cy - 0.5

    fig, ax = plt.subplots(figsize=(10, 7.5), dpi=150)
    ax.imshow(rgb_v0)

    # Find all 3D obstacle points in front of camera (p_z < -0.1)
    pts_v0_obs = view_pts_dict[0][0]  # points extracted from view 0
    if len(pts_v0_obs) > 0:
        px = pts_v0_obs[:, 0]
        py = pts_v0_obs[:, 1]
        pz = pts_v0_obs[:, 2]

        # Valid forward points in front of camera
        valid_fwd = pz < -0.2
        px_fwd = px[valid_fwd]
        py_fwd = py[valid_fwd]
        pz_fwd = pz[valid_fwd]

        z_c = -pz_fwd
        x_c = px_fwd
        y_c = camera_cfg.camera_height - py_fwd

        u_px = x_c * camera_cfg.fx / z_c + camera_cfg.cx - 0.5
        v_px = y_c * camera_cfg.fy / z_c + camera_cfg.cy - 0.5

        in_bounds = (u_px >= 0) & (u_px < w_img) & (v_px >= 0) & (v_px < h_img)
        u_valid = u_px[in_bounds]
        v_valid = v_px[in_bounds]
        h_valid = py_fwd[in_bounds]

        scatter = ax.scatter(
            u_valid,
            v_valid,
            c=h_valid,
            cmap="turbo",
            s=4,
            alpha=0.6,
            label="Chassis Obstacles (0.10m ≤ y ≤ 1.25m)",
        )
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Height y above floor datum (m)")

    # Overlay attempted targets from view 0
    target_markers = [
        ("Turn 2: (0.50, 0.50)", 0.50, 0.50, "red"),
        ("Turn 3: (0.65, 0.50)", 0.65, 0.50, "orange"),
        ("Turn 4: (0.75, 0.50)", 0.75, 0.50, "yellow"),
        ("Turn 5: (0.25, 0.50)", 0.25, 0.50, "magenta"),
    ]
    for label, u, v, col in target_markers:
        px_u = u * w_img
        py_v = v * h_img
        ax.plot(px_u, py_v, marker="x", color=col, markersize=12, markeredgewidth=3, label=label)

    # Draw the optical horizon line (v = 0.50 -> y_px = 240)
    ax.axhline(y=h_img * 0.5, color="white", linestyle="--", linewidth=1.5, label="Optical Horizon (v=0.50)")

    ax.set_title(
        f"Episode 0764 View 0: Obstacle Pointcloud Reprojection & Model Horizon Clicks\n"
        f"Notice: Clicks land on Horizon (v=0.5, y=1.25m), NOT on the floor!",
        fontsize=10,
    )
    ax.set_xlim(0, w_img)
    ax.set_ylim(h_img, 0)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)

    overlay_path = out_dir / "obstacle_overlay_view0.png"
    plt.tight_layout()
    plt.savefig(overlay_path)
    plt.close()
    print(f"Saved: {overlay_path}")

    # 4b. 2D X-Z Corridor and Scatter Plot
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)

    # Scatter obstacle points in (x, z) agent frame
    if len(all_obs_3d) > 0:
        ox = all_obs_3d[:, 0]
        oy = all_obs_3d[:, 1]
        oz = all_obs_3d[:, 2]

        in_plot = (ox >= -4.0) & (ox <= 4.0) & (oz >= -5.5) & (oz <= 2.0)
        sc = ax.scatter(
            ox[in_plot],
            oz[in_plot],
            c=oy[in_plot],
            cmap="viridis",
            s=6,
            alpha=0.7,
            label="Extracted Obstacles (colored by height y)",
        )
        cbar = plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.04)
        cbar.set_label("Obstacle Height y (m)")

    # Draw agent position and footprint circles
    ax.plot(0, 0, marker="o", color="blue", markersize=10, label="Agent Origin (0,0)")
    circ_18 = plt.Circle((0, 0), 0.18, color="blue", fill=False, linestyle="--", linewidth=2, label="Agent Radius (r=0.18m)")
    circ_10 = plt.Circle((0, 0), 0.10, color="cyan", fill=False, linestyle="-", linewidth=2, label="Habitat Radius (r=0.10m)")
    ax.add_patch(circ_18)
    ax.add_patch(circ_10)

    # Draw the attempted move vectors and corridor rectangles
    colors = ["red", "orange", "yellow", "magenta"]
    for idx, (label, u, v, col) in enumerate(target_markers):
        tres = target_eval_results[idx]
        px, py, pz = tres["agent_xyz"]
        rho_q = tres["checker_r18"]["rho_q"]
        theta_q = tres["checker_r18"]["theta_q"]

        # Draw corridor rectangle for r_a = 0.18
        theta_rad = math.radians(theta_q)
        b_x = -math.sin(theta_rad)
        b_z = -math.cos(theta_rad)
        l_x = math.cos(theta_rad)
        l_z = -math.sin(theta_rad)

        s_max = rho_q + 0.18
        r_a = 0.18
        # Corners of corridor in agent frame
        c1 = [r_a * l_x, r_a * l_z]
        c2 = [s_max * b_x + r_a * l_x, s_max * b_z + r_a * l_z]
        c3 = [s_max * b_x - r_a * l_x, s_max * b_z - r_a * l_z]
        c4 = [-r_a * l_x, -r_a * l_z]
        box = plt.Polygon([c1, c2, c3, c4], fill=False, edgecolor=col, linewidth=2, linestyle="--", label=f"Corridor: {label}")
        ax.add_patch(box)
        ax.plot(px, pz, marker="*", color=col, markersize=12)

    # Plot free simulator radial headings for rho = 1.5m
    for rec in sweep_records:
        if abs(rec["target_dist_m"] - 1.5) < 1e-4:
            th_rad = math.radians(rec["heading_deg"])
            hx = -math.sin(th_rad) * rec["sim_dist_moved"]
            hz = -math.cos(th_rad) * rec["sim_dist_moved"]
            if not rec["sim_blocked"]:
                ax.plot([0, hx], [0, hz], color="lime", alpha=0.5, linewidth=1.5)
            else:
                ax.plot([0, hx], [0, hz], color="crimson", alpha=0.3, linewidth=1)

    ax.set_title(
        f"Episode 0764 Agent X-Z Plane: Obstacles vs. 24 Headings vs. Model Corridors\n"
        f"Green lines = Simulator Actually Drove 1.5m | Red boxes = Model Failed Targets",
        fontsize=10,
    )
    ax.set_xlabel("Agent X (meters, Right = +X)")
    ax.set_ylabel("Agent Z (meters, Forward = -Z)")
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-5.0, 1.5)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.85)

    scatter_path = out_dir / "obstacle_scatter_xz.png"
    plt.tight_layout()
    plt.savefig(scatter_path)
    plt.close()
    print(f"Saved: {scatter_path}")

    # Copy to artifacts directory so user can view
    artifacts_dir = Path("C:/Users/Kruti/.gemini/antigravity/brain/845399dd-118d-4b96-a067-6f8e6f44b148")
    if artifacts_dir.exists():
        shutil.copy(overlay_path, artifacts_dir / "obstacle_overlay_view0.png")
        shutil.copy(scatter_path, artifacts_dir / "obstacle_scatter_xz.png")
        print(f"Copied plots to artifacts directory: {artifacts_dir}")

    # Save complete JSON report
    report_data = {
        "episode_id": target_ep["episode_id"],
        "scene_id": target_ep["scene_id"],
        "start_position": target_ep["start_position"],
        "target_eval_results": target_eval_results,
        "sweep_summary": {
            "total_proposals": n_total,
            "sim_blocked_count": sim_blocked_count,
            "sim_free_count": n_total - sim_blocked_count,
            "checker_w36_blocked": c18_blocked,
            "checker_w36_false_block_rate": round(false_blocks_18 / max(1, n_total - sim_blocked_count) * 100, 2),
            "checker_w20_blocked": c10_blocked,
            "checker_w20_false_block_rate": round(false_blocks_10 / max(1, n_total - sim_blocked_count) * 100, 2),
        },
        "sweep_records": sweep_records,
    }
    with open(out_dir / "episode_0764_comprehensive_diagnosis.json", "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"Saved diagnostic JSON: {out_dir / 'episode_0764_comprehensive_diagnosis.json'}")

    embodiment.close()
    print("\nDiagnostic completed successfully.")


if __name__ == "__main__":
    main()
