#!/usr/bin/env python3
"""Interactive Step Runner for AgenticNav with Antigravity / Human in the loop.

Maintains an active episode session on Habitat-Sim and persists state between steps.
Actions:
  --init <ep_idx>: Start a new episode, render initial 4 views + BEV, dump state.json.
  --act: Execute action (move_to or stop) and advance episode.
  --status: View current episode state and geodesic distance.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")
sys.path.insert(0, str(PROJECT_ROOT))

STATE_DIR = PROJECT_ROOT / "audit" / "interactive_session"
STATE_DIR.mkdir(parents=True, exist_ok=True)
SESSION_FILE = STATE_DIR / "session_state.json"

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    normalize_to_pixel,
    convert_raw_depth,
    deproject_pixel,
    camera_to_agent_frame,
    extract_view_obstacle_and_free_points,
    VIEW_AZIMUTHS_DEG,
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer


def save_image(arr: np.ndarray, path: Path):
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if len(arr.shape) == 2:
        img = Image.fromarray(arr, mode="L")
    else:
        img = Image.fromarray(arr[:, :, :3], mode="RGB")
    img.save(path)


def init_session(ep_idx: int):
    dataset_file = PROJECT_ROOT / "agenticnav_paper_replica" / "data" / "r2rce_val_unseen_17.json"
    with open(dataset_file, "r") as f:
        episodes = json.load(f)["episodes"]
        
    if ep_idx >= len(episodes):
        print(f"Error: Episode index {ep_idx} out of range (max {len(episodes)-1})")
        sys.exit(1)
        
    ep = episodes[ep_idx]
    scene_id = ep["scene_id"]
    print(f"Initializing Episode {ep_idx} ({ep['episode_id']}) in Scene {scene_id}...")
    
    embodiment = HabitatGroundEmbodiment(scene_id=scene_id, gpu_device_id=-1, max_actions=50)
    obs = embodiment.reset(ep)
    
    camera_cfg = CameraConfig()
    occ_grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
    bev_renderer = BEVRenderer()
    
    # Save visual views
    for v_idx in range(4):
        save_image(obs.rgb_views[v_idx], STATE_DIR / f"view_{v_idx}.png")
        
    ax, az, ayaw = obs.allocentric_pose
    bev_img = bev_renderer.render(occ_grid, node_poses=[(ax, az, ayaw)], current_node_idx=1)
    save_image(bev_img, STATE_DIR / "bev_map.png")
    
    initial_dist = embodiment.get_geodesic_distance_to_goal()
    shortest_path = embodiment.get_shortest_path_distance()
    
    session_data = {
        "episode_idx": ep_idx,
        "episode_id": ep["episode_id"],
        "scene_id": scene_id,
        "instruction": ep["instruction"],
        "current_node": 1,
        "actions_taken": 0,
        "initial_distance_m": initial_dist,
        "shortest_path_m": shortest_path,
        "current_distance_m": initial_dist,
        "allocentric_pose": list(obs.allocentric_pose),
        "history": [],
        "action_history": [],
        "consecutive_reselects": 0,
        "is_done": False,
        "called_stop": False,
    }
    
    with open(SESSION_FILE, "w") as f:
        json.dump(session_data, f, indent=2)
        
    prompt = f"""### Active Goal Instruction:
"{ep['instruction']}"

### Current Decision Node: 1
- Accumulated Actions Taken: 0

### Perception Frames Saved:
- Front: audit/interactive_session/view_0.png
- Right: audit/interactive_session/view_1.png
- Back:  audit/interactive_session/view_2.png
- Left:  audit/interactive_session/view_3.png
- BEV:   audit/interactive_session/bev_map.png

### Next Action Required:
Inspect views and issue move_to or stop.
"""
    with open(STATE_DIR / "current_prompt.txt", "w") as f:
        f.write(prompt)
        
    print(f"\nEpisode Initialized Successfully!")
    print(prompt)
    embodiment.close()


def execute_action(action_name: str, view: int = 0, u: float = 0.5, v: float = 0.5, reason: str = "", sim_fallback: bool = False):
    if not SESSION_FILE.exists():
        print("Error: No active session. Run --init <idx> first.")
        sys.exit(1)
        
    with open(SESSION_FILE, "r") as f:
        sess = json.load(f)
        
    if sess["is_done"]:
        print("Episode is already finished.")
        return
        
    dataset_file = PROJECT_ROOT / "agenticnav_paper_replica" / "data" / "r2rce_val_unseen_17.json"
    with open(dataset_file, "r") as f:
        episodes = json.load(f)["episodes"]
    ep = episodes[sess["episode_idx"]]
    
    embodiment = HabitatGroundEmbodiment(scene_id=sess["scene_id"], gpu_device_id=-1, max_actions=50)
    obs = embodiment.reset(ep)
    
    # Replay past successful actions
    for past_act in sess["action_history"]:
        obs = embodiment.step_discrete(
            turn_steps_15deg=past_act["turn_steps"],
            forward_steps_025m=past_act["forward_steps"],
        )
        
    if action_name == "stop":
        dist = embodiment.get_geodesic_distance_to_goal()
        success = (dist <= 3.0)
        spl = (sess["shortest_path_m"] / max(sess["shortest_path_m"], sess["actions_taken"] * 0.25)) if success else 0.0
        
        sess["called_stop"] = True
        sess["is_done"] = True
        sess["success"] = success
        sess["spl"] = spl
        sess["current_distance_m"] = dist
        sess["history"].append(f"Node {sess['current_node']}: STOP ('{reason}') -> Final Distance = {dist:.2f}m")
        
        with open(SESSION_FILE, "w") as f:
            json.dump(sess, f, indent=2)
            
        print("\n=======================================================")
        print(f"EPISODE FINISHED: STOP CALLED")
        print(f"  Success: {'SUCCESS' if success else 'FAILED'} (Distance to goal: {dist:.2f} m)")
        print(f"  SPL: {spl:.4f}")
        print(f"  Actions taken: {sess['actions_taken']}")
        print(f"  Reason: {reason}")
        print("=======================================================")
        embodiment.close()
        return
        
    # Evaluate move_to
    camera_cfg = CameraConfig()
    checker = GroundCorridorChecker()
    
    # Backproject depth at (u, v) in chosen view
    d_img = obs.depth_views[view]
    x0, y0 = normalize_to_pixel(u, v, camera_cfg)
    z_raw = float(d_img[y0, x0])
    
    is_valid, z_c, error_cause = convert_raw_depth(z_raw, camera_cfg)
    if not is_valid:
        print(f"Reselect: Invalid depth at ({u}, {v}): {z_raw:.2f} m ({error_cause})")
        embodiment.close()
        return
        
    # Canonical camera frame to agent frame using verified pinhole module
    p_c = deproject_pixel(x0, y0, z_c, camera_cfg, is_normalized=False)
    p_agent = camera_to_agent_frame(p_c, view, camera_cfg)
    p_x_agent = float(p_agent[0])
    p_z_agent = float(p_agent[2])
    
    # Extract 2D obstacles for corridor checking
    obs_list = []
    for vi, dimg in obs.depth_views.items():
        pts_obs, _ = extract_view_obstacle_and_free_points(dimg, view_idx=vi, config=camera_cfg, stride=4, y_min=0.10, y_max=1.25)
        if len(pts_obs) > 0:
            obs_list.append(pts_obs[:, [0, 2]])
    obstacles_2d = np.vstack(obs_list) if obs_list else np.empty((0, 2))
    
    check_res = checker.evaluate(p_x=p_x_agent, p_z=p_z_agent, obstacles_2d=obstacles_2d)
    
    if check_res.status == "reselect":
        print(f"\n[CORRIDOR CHECK REJECTED]: status={check_res.status}, cause={check_res.cause}")
        print(f"  Target raw dist: {check_res.raw_distance_m:.2f}m, bearing: {check_res.raw_bearing_deg:.1f} deg")
        print(f"  Collisions detected: {check_res.collision_point_count}")
        if not sim_fallback:
            embodiment.close()
            return
        print(f"  [SIM-FALLBACK]: Bypassing synthetic 2D corridor rejection and testing in Habitat-Sim...")
        # Compute kinematic steps directly towards target
        D_raw, theta_deg, rho_q, theta_q_deg = checker.compute_kinematics(p_x_agent, p_z_agent)
        turn_steps = int(round(theta_q_deg / checker.turn_step_deg))
        forward_steps = int(round(rho_q / checker.forward_step_m))
        if turn_steps == 0 and forward_steps == 0:
            print("  [SIM-FALLBACK]: Computed 0 motion. Aborting step.")
            embodiment.close()
            return
        # Override check_res with kinematic motion
        check_res.turn_steps = turn_steps
        check_res.forward_steps = forward_steps
        check_res.quantized_theta_deg = theta_q_deg
        check_res.quantized_rho_m = rho_q
        
    # Execute step in simulator
    print(f"\n[EXECUTING MOTION]: turns={check_res.turn_steps} x 15 deg ({check_res.quantized_theta_deg:.1f} deg), forward={check_res.forward_steps} x 0.25 m ({check_res.quantized_rho_m:.2f} m)")
    obs = embodiment.step_discrete(
        turn_steps_15deg=check_res.turn_steps,
        forward_steps_025m=check_res.forward_steps,
    )
    
    sess["actions_taken"] += (abs(check_res.turn_steps) + check_res.forward_steps)
    sess["current_node"] += 1
    new_dist = embodiment.get_geodesic_distance_to_goal()
    sess["current_distance_m"] = new_dist
    sess["allocentric_pose"] = list(obs.allocentric_pose)
    sess["action_history"].append({
        "turn_steps": check_res.turn_steps,
        "forward_steps": check_res.forward_steps,
        "view": view, "u": u, "v": v, "reason": reason
    })
    sess["history"].append(f"Node {sess['current_node']}: move_to(view={view}, u={u:.2f}, v={v:.2f}) -> Dist = {new_dist:.2f}m")
    
    # Save updated visual views
    for v_idx in range(4):
        save_image(obs.rgb_views[v_idx], STATE_DIR / f"view_{v_idx}.png")
        
    occ_grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
    bev_renderer = BEVRenderer()
    ax, az, ayaw = obs.allocentric_pose
    bev_img = bev_renderer.render(occ_grid, node_poses=[(ax, az, ayaw)], current_node_idx=sess["current_node"])
    save_image(bev_img, STATE_DIR / "bev_map.png")
    
    with open(SESSION_FILE, "w") as f:
        json.dump(sess, f, indent=2)
        
    prompt = f"""### Active Goal Instruction:
"{sess['instruction']}"

### Current Decision Node: {sess['current_node']}
- Accumulated Actions Taken: {sess['actions_taken']}

### Perception Frames Updated:
- Front: audit/interactive_session/view_0.png
- Right: audit/interactive_session/view_1.png
- Back:  audit/interactive_session/view_2.png
- Left:  audit/interactive_session/view_3.png
- BEV:   audit/interactive_session/bev_map.png
"""
    with open(STATE_DIR / "current_prompt.txt", "w") as f:
        f.write(prompt)
        
    print(f"\nStep Executed Successfully! Now at Node {sess['current_node']}")
    print(f"Current distance to goal: {new_dist:.2f} m")
    embodiment.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=int, default=None, help="Episode index to initialize")
    parser.add_argument("--act", action="store_true", help="Execute action")
    parser.add_argument("--action", type=str, default="move_to", choices=["move_to", "stop"])
    parser.add_argument("--view", type=int, default=0)
    parser.add_argument("--u", type=float, default=0.5)
    parser.add_argument("--v", type=float, default=0.5)
    parser.add_argument("--reason", type=str, default="")
    parser.add_argument("--sim-fallback", action="store_true", help="Fallback to simulator execution if 2D corridor checker blocks")
    parser.add_argument("--status", action="store_true", help="Print session status")
    args = parser.parse_args()
    
    if args.init is not None:
        init_session(args.init)
    elif args.act:
        execute_action(
            action_name=args.action,
            view=args.view,
            u=args.u,
            v=args.v,
            reason=args.reason,
            sim_fallback=args.sim_fallback,
        )
    elif args.status:
        if SESSION_FILE.exists():
            with open(SESSION_FILE) as f:
                print(f.read())
        else:
            print("No active session.")

if __name__ == "__main__":
    main()
