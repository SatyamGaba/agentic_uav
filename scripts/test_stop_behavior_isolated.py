#!/usr/bin/env python3
"""Check G2.7 & G9.2 Extension: Stop Behavior in Isolation.

Tests the VLM (qwen3.5:4b) when placed ~1.0m from the goal:
1. With legacy prompt (< 6.0m prohibition).
2. With clean milestone prompt (no artificial distance prohibition).
3. With clean milestone prompt + thinking enabled (thinking=True).
4. With explicit arrival milestone prompt.

Outputs detailed audit logs and JSON to audit/phase9/stop_isolation_battery.json.
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.harness.tools import get_available_tools
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points


CLEAN_SYSTEM_PROMPT = """You are an autonomous robot navigation agent. Your objective is to follow natural language navigation instructions to reach the target destination in an indoor environment.

### Perception and Sensor Setup:
At each decision step, you are provided with:
1. **Four panoramic camera views**:
   - View 0: Front (0°)
   - View 1: Right (-90°)
   - View 2: Back (180°)
   - View 3: Left (+90°)
   Normalized image coordinates (u, v) are in [0.0, 1.0], where (0.0, 0.0) is the top-left corner and (1.0, 1.0) is the bottom-right corner.
2. **Bird's-Eye View (BEV) Map**:
   - White: Explored free space.
   - Black: Obstacles and walls.
   - Cyan line: Past trajectory.
   - Red circle with directional arrows: Current position and heading.

### Available Tools:
1. `query_depth(points)`: Inspect physical metric distance and 3D clearance for candidate targets in meters.
2. `recall(node, view)`: Inspect past camera views from any previously visited node.
3. `move_to(view, u, v, reason)`: Navigate towards the point (u, v) seen in view.
4. `stop(reason)`: Declare arrival at the final goal destination.

### MANDATORY NAVIGATION & STOPPING RULES:
- Deconstruct the goal instruction into sequential milestones.
- Keep moving along corridors, doorways, or open floor until all milestones are satisfied.
- When you have completed all navigation milestones and reached the target room/object indicated by the instruction, invoke `stop`.
"""

LEGACY_SYSTEM_PROMPT = """You are an autonomous robot navigation agent. Your objective is to follow natural language navigation instructions to reach the target destination in an indoor environment.

### Available Tools:
1. `query_depth(points)`: Inspect physical metric distance and 3D clearance for candidate targets in meters.
2. `recall(node, view)`: Inspect past camera views from any previously visited node.
3. `move_to(view, u, v, reason)`: Navigate towards the point (u, v) seen in view.
4. `stop(reason)`: Declare arrival at the final goal destination.

### MANDATORY NAVIGATION & STOPPING RULES:
- **Instruction Milestones**: Break the instruction down into sequential milestones.
- **Do NOT Stop Prematurely**: Real indoor navigation routes typically require 8 to 15 meters of movement across several waypoints. You must NEVER call `stop` after only 1-3 moves or under 6.0 meters of travel.
- **OBSTACLE CLEARANCE IS NOT LANDMARK ARRIVAL**: Facing a wall or low clearance means you must TURN, NOT stop.
- Only call `stop` when ALL sequential clauses in the instruction have been completely fulfilled and you are standing directly in front of the final target landmark.
"""


def main():
    print("=== CHECK 2: STOP BEHAVIOR IN ISOLATION ===")
    audit_dir = ROOT_DIR / "audit" / "phase9" / "stop_isolation"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    episodes_file = ROOT_DIR / "agenticnav_paper_replica" / "data" / "episode_list_100.json"
    with open(episodes_file, "r") as f:
        all_eps = json.load(f)["episodes"]
    episodes = [e for e in all_eps if e.get("scene_id") == "gZ6f7yhEvPG"]
    
    scene_id = "gZ6f7yhEvPG"
    print(f"Initializing Habitat-Sim on scene {scene_id}...")
    embodiment = HabitatGroundEmbodiment(scene_id=scene_id, gpu_device_id=-1, max_actions=50)
    
    # Select test episode: Episode 2 (index 1) in gZ6f7yhEvPG
    ep = episodes[1]
    instruction = ep["instruction"]
    start_pos = ep["start_position"]
    start_rot = ep["start_rot"] if "start_rot" in ep else ep["start_rotation"]
    goal_pos = ep["goal_position"]
    
    print(f"\nTest Episode: {ep['episode_id']}")
    print(f"Instruction: \"{instruction}\"")
    print(f"Goal position: {goal_pos}")
    
    # Reset embodiment to episode
    obs = embodiment.reset(ep)
    
    # Use snapped goal from embodiment
    pf = embodiment.sim.pathfinder
    goal_pos = embodiment.current_goal_pos
    start_pos = embodiment.sim.get_agent(0).get_state().position
    print(f"Agent start pos: {start_pos}, goal pos: {goal_pos}")
    
    # Teleport agent directly to near-goal position (within ~0.8m)
    # Using shortest path from start to goal to find the waypoint right before goal
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start_pos, dtype=np.float32)
    path.requested_end = np.array(goal_pos, dtype=np.float32)
    found = pf.find_path(path)
    
    pts = path.points
    print(f"Shortest path found: {found}, has {len(pts)} points, total length = {path.geodesic_distance:.2f}m")
    
    # Pick a point along path ~0.8m before goal
    test_pos = pts[-1]
    goal_arr = np.array(goal_pos, dtype=np.float32)
    for p in reversed(pts):
        d = float(np.linalg.norm(np.array(p) - goal_arr))
        if 0.4 <= d <= 1.2:
            test_pos = p
            break
            
    print(f"Selected test teleport position: {test_pos}, distance to goal: {float(np.linalg.norm(np.array(test_pos) - goal_arr)):.2f}m")
    
    # Teleport agent
    agent = embodiment.sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = np.array(test_pos, dtype=np.float32)
    # Face towards goal
    dx = float(goal_pos[0] - test_pos[0])
    dz = float(goal_pos[2] - test_pos[2])
    yaw = math.atan2(-dx, -dz)
    import quaternion
    agent_state.rotation = quaternion.from_euler_angles(0.0, yaw, 0.0)
    agent.set_state(agent_state)
    
    # Fetch sensor observation at isolated position
    obs = embodiment.step_discrete(turn_steps_15deg=0, forward_steps_025m=0) # refresh buffers
    dist_to_goal = embodiment.get_geodesic_distance_to_goal()
    print(f"Agent state established: current geodesic dist to goal = {dist_to_goal:.2f}m")
    
    # Build BEV map
    camera_cfg = CameraConfig()
    occ_grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
    bev_renderer = BEVRenderer()
    
    obstacle_pts_list = []
    for v_idx in range(4):
        d_img = obs.depth_views[v_idx]
        pts_obs, _ = extract_view_obstacle_and_free_points(d_img, view_idx=v_idx, config=camera_cfg, stride=4, y_min=0.10, y_max=1.25)
        if len(pts_obs) > 0:
            obstacle_pts_list.append(pts_obs)
            
    ax, az, ayaw = obs.allocentric_pose
    if obstacle_pts_list:
        cur_obs_xyz = np.vstack(obstacle_pts_list)
        ox = ax + math.cos(ayaw)*cur_obs_xyz[:,0] + math.sin(ayaw)*cur_obs_xyz[:,2]
        oz = az - math.sin(ayaw)*cur_obs_xyz[:,0] + math.cos(ayaw)*cur_obs_xyz[:,2]
        obs_world = np.column_stack([ox, oz])
    else:
        obs_world = np.empty((0, 2))
        
    occ_grid.update(agent_x_m=ax, agent_z_m=az, obstacle_points_world_xz=obs_world)
    bev_img = bev_renderer.render(occ_grid, node_poses=[(ax, az, ayaw)], current_node_idx=1)
    
    tools = get_available_tools(enable_depth=True, enable_recall=False)
    
    # Define test conditions
    conditions = [
        {
            "name": "Cond1_LegacyPrompt_ThinkingFalse",
            "thinking": False,
            "system_prompt": LEGACY_SYSTEM_PROMPT,
            "user_prompt": (
                f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
                f"### Current State:\n"
                f"- Active Decision Node: 3\n"
                f"- Accumulated Distance Traversed: 2.80 meters\n\n"
                f"> [NAVIGATION NOTICE]: You have traversed only 2.80m. Typical target destinations require 8 to 15 meters. Do NOT stop now; continue following the route along the hallway.\n\n"
                f"### Action History (Recent Steps):\n"
                f"Node 1: move_to(view=0, u=0.5, v=0.6) [Advanced down corridor]\n"
                f"Node 2: move_to(view=0, u=0.5, v=0.5) [Passed through double doors]\n\n"
                f"Decide on your next action now."
            ),
        },
        {
            "name": "Cond2_CleanPrompt_ThinkingFalse",
            "thinking": False,
            "system_prompt": CLEAN_SYSTEM_PROMPT,
            "user_prompt": (
                f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
                f"### Current State:\n"
                f"- Active Decision Node: 3\n"
                f"- Accumulated Distance Traversed: 2.80 meters\n\n"
                f"### Action History (Recent Steps):\n"
                f"Node 1: move_to(view=0, u=0.5, v=0.6) [Exited bedroom]\n"
                f"Node 2: move_to(view=0, u=0.5, v=0.5) [Passed through double doors into hallway]\n\n"
                f"### Instructions for this Step:\n"
                f"1. Deconstruct the instruction milestones.\n"
                f"2. Inspect the camera views to see if you have arrived at the final target.\n"
                f"3. If all milestones are satisfied, call `stop`. Otherwise `move_to`.\n"
                f"Decide on your next action now."
            ),
        },
        {
            "name": "Cond3_CleanPrompt_ThinkingTrue",
            "thinking": True,
            "system_prompt": CLEAN_SYSTEM_PROMPT,
            "user_prompt": (
                f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
                f"### Current State:\n"
                f"- Active Decision Node: 3\n"
                f"- Accumulated Distance Traversed: 2.80 meters\n\n"
                f"### Action History (Recent Steps):\n"
                f"Node 1: move_to(view=0, u=0.5, v=0.6) [Exited bedroom]\n"
                f"Node 2: move_to(view=0, u=0.5, v=0.5) [Passed through double doors into hallway]\n\n"
                f"### Instructions for this Step:\n"
                f"1. Deconstruct the instruction milestones.\n"
                f"2. Inspect the camera views to see if you have arrived at the final target.\n"
                f"3. If all milestones are satisfied, call `stop`. Otherwise `move_to`.\n"
                f"Decide on your next action now."
            ),
        },
        {
            "name": "Cond4_ExplicitArrivalMilestone_ThinkingFalse",
            "thinking": False,
            "system_prompt": CLEAN_SYSTEM_PROMPT,
            "user_prompt": (
                f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
                f"### Current State:\n"
                f"- Active Decision Node: 3\n"
                f"- Milestones Completed: [1. Exit bedroom: DONE, 2. Through double doors: DONE, 3. In hallway: REACHED]\n\n"
                f"### Action History (Recent Steps):\n"
                f"Node 1: Exited bedroom\n"
                f"Node 2: Went through double doors and entered the target hallway\n\n"
                f"All milestones of the instruction have been completely satisfied. Please declare arrival.\n"
                f"Decide on your next action now."
            ),
        },
    ]
    
    results = []
    
    for cond in conditions:
        print(f"\n--- Running Condition: {cond['name']} ---")
        client = OllamaNavClient(
            model_name="qwen3.5:4b",
            thinking=cond["thinking"],
            temperature=0.0,
            timeout_s=60.0,
        )
        
        messages = [
            {"role": "system", "content": cond["system_prompt"]},
            {
                "role": "user",
                "content": cond["user_prompt"],
                "rgb_views": obs.rgb_views,
                "bev_image": bev_img,
            },
        ]
        
        t0 = time.time()
        try:
            resp = client(messages=messages, tools=tools)
            dt = time.time() - t0
            raw_text = resp.get("raw_text", "")
            thinking_content = resp.get("thinking_content", "")
            tool_calls = resp.get("tool_calls", [])
            
            tool_name = tool_calls[0].get("name") if tool_calls else "NO_TOOL_CALL"
            tool_args = tool_calls[0].get("arguments") if tool_calls else {}
            
            called_stop = (tool_name == "stop")
            
            print(f"Elapsed: {dt:.2f}s | Called tool: {tool_name} | Called STOP? {called_stop}")
            print(f"Tool args: {tool_args}")
            if thinking_content:
                print(f"Thinking summary (first 150 chars): {thinking_content[:150]}...")
            print(f"Model message: {raw_text[:200]}")
            
            res_entry = {
                "condition": cond["name"],
                "thinking_enabled": cond["thinking"],
                "called_stop": called_stop,
                "tool_name": tool_name,
                "tool_args": tool_args,
                "raw_text": raw_text,
                "thinking_content": thinking_content,
                "latency_s": dt,
            }
            results.append(res_entry)
            
        except Exception as e:
            print(f"ERROR executing {cond['name']}: {e}")
            results.append({
                "condition": cond["name"],
                "error": str(e),
                "called_stop": False,
            })
            
    # Save results
    output_path = audit_dir / "stop_isolation_battery.json"
    with open(output_path, "w") as f:
        json.dump({
            "episode_id": ep["episode_id"],
            "instruction": instruction,
            "geodesic_distance_to_goal_m": dist_to_goal,
            "test_conditions": results,
        }, f, indent=2)
        
    print(f"\nAudit results successfully saved to {output_path}")
    embodiment.close()

if __name__ == "__main__":
    main()
