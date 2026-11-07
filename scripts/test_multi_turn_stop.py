#!/usr/bin/env python3
"""Check G2.7 & G9.2 Extension Part 2: Multi-Turn Stop Decision in Isolation.

Tests Turn 2 (after query_depth feedback) and obstacle reselect feedback:
Does qwen3.5:4b call stop() once depth or reselect feedback reveals wall / terminal landmark?
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.harness.tools import get_available_tools
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.harness.executor import ToolExecutor


CLEAN_SYSTEM_PROMPT = """You are an autonomous robot navigation agent. Your objective is to follow natural language navigation instructions to reach the target destination in an indoor environment.

### Perception and Sensor Setup:
At each decision step, you are provided with:
1. Four panoramic camera views (0: Front, 1: Right, 2: Back, 3: Left).
2. Bird's-Eye View (BEV) Map.

### Available Tools:
1. `query_depth(points)`: Inspect physical metric distance and 3D clearance for candidate targets in meters.
2. `move_to(view, u, v, reason)`: Navigate towards the point (u, v) seen in view.
3. `stop(reason)`: Declare arrival at the final goal destination.

### MANDATORY NAVIGATION & STOPPING RULES:
- Deconstruct the goal instruction into sequential milestones.
- Keep moving along corridors, doorways, or open floor until all milestones are satisfied.
- When you have completed all navigation milestones and reached the target room/object indicated by the instruction, invoke `stop`.
"""

LEGACY_SYSTEM_PROMPT = """You are an autonomous robot navigation agent. Your objective is to follow natural language navigation instructions to reach the target destination in an indoor environment.

### Available Tools:
1. `query_depth(points)`: Inspect physical metric distance in meters.
2. `move_to(view, u, v, reason)`: Navigate towards the point (u, v) seen in view.
3. `stop(reason)`: Declare arrival at the final goal destination.

### MANDATORY NAVIGATION & STOPPING RULES:
- **Instruction Milestones**: Break the instruction down into sequential milestones.
- **Do NOT Stop Prematurely**: Real indoor navigation routes typically require 8 to 15 meters of movement across several waypoints. You must NEVER call `stop` after only 1-3 moves or under 6.0 meters of travel.
- **OBSTACLE CLEARANCE IS NOT LANDMARK ARRIVAL**: Facing a wall or low clearance means you must TURN, NOT stop.
- Only call `stop` when ALL sequential clauses in the instruction have been completely fulfilled and you are standing directly in front of the final target landmark.
"""


def main():
    print("=== MULTI-TURN STOP DECISION TEST IN ISOLATION ===")
    audit_dir = ROOT_DIR / "audit" / "phase9" / "stop_isolation"
    audit_dir.mkdir(parents=True, exist_ok=True)
    
    episodes_file = ROOT_DIR / "agenticnav_paper_replica" / "data" / "episode_list_100.json"
    with open(episodes_file, "r") as f:
        all_eps = json.load(f)["episodes"]
    episodes = [e for e in all_eps if e.get("scene_id") == "gZ6f7yhEvPG"]
    
    scene_id = "gZ6f7yhEvPG"
    embodiment = HabitatGroundEmbodiment(scene_id=scene_id, gpu_device_id=-1, max_actions=50)
    
    ep = episodes[1]
    instruction = ep["instruction"]
    obs = embodiment.reset(ep)
    
    pf = embodiment.sim.pathfinder
    goal_pos = embodiment.current_goal_pos
    start_pos = embodiment.sim.get_agent(0).get_state().position
    
    import habitat_sim
    path = habitat_sim.ShortestPath()
    path.requested_start = np.array(start_pos, dtype=np.float32)
    path.requested_end = np.array(goal_pos, dtype=np.float32)
    pf.find_path(path)
    pts = path.points
    
    # Teleport to 0.55m from goal
    test_pos = pts[-1]
    goal_arr = np.array(goal_pos, dtype=np.float32)
    for p in reversed(pts):
        d = float(np.linalg.norm(np.array(p) - goal_arr))
        if 0.4 <= d <= 1.2:
            test_pos = p
            break
            
    agent = embodiment.sim.get_agent(0)
    agent_state = agent.get_state()
    agent_state.position = np.array(test_pos, dtype=np.float32)
    dx = float(goal_pos[0] - test_pos[0])
    dz = float(goal_pos[2] - test_pos[2])
    yaw = math.atan2(-dx, -dz)
    import quaternion
    agent_state.rotation = quaternion.from_euler_angles(0.0, yaw, 0.0)
    agent.set_state(agent_state)
    
    obs = embodiment.step_discrete(turn_steps_15deg=0, forward_steps_025m=0)
    dist_to_goal = embodiment.get_geodesic_distance_to_goal()
    print(f"Agent state established: current geodesic dist to goal = {dist_to_goal:.2f}m")
    
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
    
    executor = ToolExecutor(camera_config=camera_cfg, corridor_checker=None, image_cache=None)
    
    # We test Turn 2: Simulate Turn 1 query_depth response
    # Query depth points in front view (view 0)
    query_pts = [{"u": 0.5, "v": 0.6, "view": 0}, {"u": 0.5, "v": 0.5, "view": 1}]
    depth_res = executor.execute_query_depth(query_pts, obs.depth_views)
    print(f"Executed real depth reading at goal: {depth_res}")
    
    # Condition A: Turn 2 with Legacy Prompt (accumulated < 6.0m lock)
    legacy_turn1_msg = (
        f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
        f"### Current State:\n"
        f"- Active Decision Node: 3\n"
        f"- Accumulated Distance Traversed: 2.80 meters\n\n"
        f"> [NAVIGATION NOTICE]: You have traversed only 2.80m. Typical target destinations require 8 to 15 meters. Do NOT stop now; continue following the route along the hallway.\n\n"
        f"### Action History (Recent Steps):\n"
        f"Node 1: move_to(view=0, u=0.5, v=0.6) [Exited bedroom]\n"
        f"Node 2: move_to(view=0, u=0.5, v=0.5) [Entered kitchen hallway]\n\n"
        f"Decide on your next action now."
    )
    
    # Condition B: Turn 2 with Clean Prompt (no distance lock)
    clean_turn1_msg = (
        f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
        f"### Current State:\n"
        f"- Active Decision Node: 3\n"
        f"- Accumulated Distance Traversed: 2.80 meters\n\n"
        f"### Action History (Recent Steps):\n"
        f"Node 1: move_to(view=0, u=0.5, v=0.6) [Exited bedroom]\n"
        f"Node 2: move_to(view=0, u=0.5, v=0.5) [Entered kitchen hallway near counter]\n\n"
        f"### Instructions for this Step:\n"
        f"Inspect the camera views and depth readings. If all instruction milestones are fulfilled and you have reached the destination, call `stop`. Otherwise call `move_to`."
    )
    
    # Condition C: Reselect feedback when blocked by terminal counter/wall
    blocked_feedback_msg = (
        f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
        f"### Current State:\n"
        f"- Active Decision Node: 3\n"
        f"- Accumulated Distance Traversed: 2.80 meters\n\n"
        f"### Action History (Recent Steps):\n"
        f"Node 1: Exited bedroom\n"
        f"Node 2: Entered kitchen hallway\n\n"
        f"### Tool Feedback from move_to:\n"
        f"{{\"status\": \"reselect\", \"cause\": \"no_motion\", \"clearance_m\": 0.20, \"message\": \"Forward motion is blocked by a counter/wall 0.50m ahead. If this is your destination (counter in kitchen/hallway), call stop(). Otherwise, turn towards open corridor.\"}}\n\n"
        f"Decide on your next action now."
    )

    test_runs = [
        {
            "name": "Turn2_LegacyPrompt_DepthFeedback",
            "system": LEGACY_SYSTEM_PROMPT,
            "messages": [
                {"role": "system", "content": LEGACY_SYSTEM_PROMPT},
                {"role": "user", "content": legacy_turn1_msg, "rgb_views": obs.rgb_views, "bev_image": bev_img},
                {"role": "assistant", "content": json.dumps({"name": "query_depth", "arguments": {"points": query_pts}})},
                {"role": "tool", "name": "query_depth", "content": json.dumps(depth_res)},
            ],
            "tools": get_available_tools(enable_depth=False, enable_recall=False),
        },
        {
            "name": "Turn2_CleanPrompt_DepthFeedback",
            "system": CLEAN_SYSTEM_PROMPT,
            "messages": [
                {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
                {"role": "user", "content": clean_turn1_msg, "rgb_views": obs.rgb_views, "bev_image": bev_img},
                {"role": "assistant", "content": json.dumps({"name": "query_depth", "arguments": {"points": query_pts}})},
                {"role": "tool", "name": "query_depth", "content": json.dumps(depth_res)},
            ],
            "tools": get_available_tools(enable_depth=False, enable_recall=False),
        },
        {
            "name": "Turn2_BlockedReselectFeedback",
            "system": CLEAN_SYSTEM_PROMPT,
            "messages": [
                {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
                {"role": "user", "content": blocked_feedback_msg, "rgb_views": obs.rgb_views, "bev_image": bev_img},
            ],
            "tools": get_available_tools(enable_depth=False, enable_recall=False),
        },
    ]

    results = []
    client = OllamaNavClient(model_name="qwen3.5:4b", thinking=False, temperature=0.0)

    for tr in test_runs:
        print(f"\n--- Testing {tr['name']} ---")
        t0 = time.time()
        res = client(tr["messages"], tr["tools"])
        dt = time.time() - t0
        tcalls = res.get("tool_calls", [])
        tname = tcalls[0].get("name") if tcalls else "NONE"
        targs = tcalls[0].get("arguments") if tcalls else {}
        called_stop = (tname == "stop")
        
        print(f"Elapsed: {dt:.2f}s | Tool: {tname} | STOP? {called_stop}")
        print(f"Args: {targs}")
        results.append({
            "test_name": tr["name"],
            "tool_name": tname,
            "called_stop": called_stop,
            "tool_args": targs,
            "latency_s": dt,
        })

    out_file = audit_dir / "multi_turn_stop_battery.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_file}")
    embodiment.close()

if __name__ == "__main__":
    main()
