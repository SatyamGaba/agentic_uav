"""
Comprehensive End-to-End Robustness and Sanity Verification Suite for AgenticNav.
Executes sample inputs through every single module, inspects outputs, and validates edge cases.
"""
import sys
import os
import math
import numpy as np
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    deproject_pixel,
    reproject_point,
    camera_to_agent_frame,
    convert_raw_depth,
    extract_depth_patch,
    unproject_depth_image,
    VIEW_AZIMUTHS_DEG,
    VIEW_LABELS,
)
from agenticnav_paper_replica.geometry.safety_region import (
    GroundCorridorChecker,
    SafetyCheckResult,
)
from agenticnav_paper_replica.mapping.occupancy_grid import OccupancyGrid2D
from agenticnav_paper_replica.mapping.bev_renderer import BEVRenderer
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache
from agenticnav_paper_replica.memory.history import ActionHistory, HistoryRecord
from agenticnav_paper_replica.harness.tools import get_available_tools
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.harness.step_loop import StepDecisionLoop, StepLoopResult
from agenticnav_paper_replica.harness.runner import run_episode, EpisodeRunResult
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
from agenticnav_paper_replica.evaluation.metrics import compute_episode_metrics, EpisodeMetrics
from agenticnav_paper_replica.evaluation.failure_classifier import classify_failure
from agenticnav_paper_replica.evaluation.bootstrap import compute_bootstrap_ci, BootstrapResult
from agenticnav_paper_replica.traces.trace_logger import TraceLogger


def test_section_1_geometry_and_pinhole():
    print("\n" + "=" * 70)
    print("MODULE 1: PINHOLE DEPROJECTION & SPATIAL GEOMETRY (geometry/pinhole.py)")
    print("=" * 70)

    cfg = CameraConfig()
    print(f"Camera Configuration: {cfg.image_width}x{cfg.image_height}, HFOV: {cfg.hfov_deg}°, Height: {cfg.camera_height}m")
    print(f"Intrinsics: fx={cfg.fx:.2f}, fy={cfg.fy:.2f}, cx={cfg.cx:.2f}, cy={cfg.cy:.2f}")

    # Case A: Center pixel in View 0 (Front) at 2.5m depth
    p_cam = deproject_pixel(320, 240, 2.5, cfg, is_normalized=False)
    p_rob = camera_to_agent_frame(p_cam, view_idx=0, config=cfg)
    print(f"Sample 1 [Center View 0]: (x=320, y=240, z=2.5m)")
    print(f"  -> Camera Frame: {p_cam.round(3).tolist()}")
    print(f"  -> Robot Frame : {p_rob.round(3).tolist()} (Expected X~0, Y~1.25, Z=-2.5)")
    assert abs(p_rob[0]) < 0.01 and abs(p_rob[2] - (-2.5)) < 0.01, "View 0 center mapping failed"

    # Case B: 4-View Azimuth Rotation Check
    for v_idx in [0, 1, 2, 3]:
        p_c = deproject_pixel(320, 240, 3.0, cfg, is_normalized=False)
        p_r = camera_to_agent_frame(p_c, view_idx=v_idx, config=cfg)
        lbl = VIEW_LABELS[v_idx]
        print(f"  -> View {v_idx} ({lbl}) Center: Robot point = [X={p_r[0]:.2f}, Z={p_r[2]:.2f}]")
        if v_idx == 0:
            assert abs(p_r[0]) < 0.01 and p_r[2] < -2.99
        elif v_idx == 1:
            assert p_r[0] > 2.99 and abs(p_r[2]) < 0.01
        elif v_idx == 2:
            assert abs(p_r[0]) < 0.01 and p_r[2] > 2.99
        elif v_idx == 3:
            assert p_r[0] < -2.99 and abs(p_r[2]) < 0.01

    # Case C: Depth Validity and Out-of-Range Handling
    valid, d_c, cause = convert_raw_depth(3.5, cfg)
    assert valid and cause is None, "Valid depth failed"
    valid, d_c, cause = convert_raw_depth(6.5, cfg)
    assert not valid and cause == "out_of_range", f"Expected out_of_range for 6.5m, got {cause}"
    valid, d_c, cause = convert_raw_depth(np.nan, cfg)
    assert not valid and cause == "invalid_depth", f"Expected invalid_depth for NaN, got {cause}"
    valid, d_c, cause = convert_raw_depth(-0.5, cfg)
    assert not valid and cause == "invalid_depth", f"Expected invalid_depth for -0.5, got {cause}"
    print(f"Sample 2 [Depth Validity]: 3.5m -> VALID, 6.5m -> out_of_range, NaN -> invalid_depth, -0.5m -> invalid_depth")
    print("[PASS] Module 1 verified cleanly.")


def test_section_2_safety_corridor_and_kinematics():
    print("\n" + "=" * 70)
    print("MODULE 2: SAFETY CORRIDOR & KINEMATICS QUANTIZATION (geometry/safety_region.py)")
    print("=" * 70)

    checker = GroundCorridorChecker(
        robot_radius=0.18,      # r_a [P]
        base_margin=0.30,       # m [?]
        rho_max=2.50,           # rho_max [?]
        turn_step_deg=15.0,     # discrete primitives [P]
        forward_step_m=0.25,    # discrete primitives [P]
        zero_rho_policy="rotate",
        allow_adjacent_bin_snap=False,
    )

    print(f"Corridor Configuration: r_a={checker.r_a}m, margin={checker.base_margin}m, rho_max={checker.rho_max}m")
    print(f"Corridor Dimensions: Lateral Bound |h| < {checker.r_a}m (Total Width = {2*checker.r_a}m)")

    # Test 1: Clear Straight Path
    D_raw, th_deg, rho_q, th_q = checker.compute_kinematics(p_x=0.0, p_z=-2.0)
    print(f"Sample 1 [Target Ahead: X=0.0, Z=-2.0m]:")
    print(f"  -> D_raw = {D_raw:.2f}m, th_deg = {th_deg:.1f}°")
    print(f"  -> Standoff applied (D - 0.3m = 1.7m) -> Quantized rho_q = {rho_q:.2f}m, th_q = {th_q:.1f}°")
    assert rho_q == 1.50, f"Expected floor(1.7/0.25)*0.25 = 1.50, got {rho_q}"
    assert th_q == 0.0

    # Test 2: Obstacle right in the corridor (|h| < r_a = 0.18m)
    obstacles_in_path = np.array([[0.05, -1.0]])  # h=0.05 < 0.18, s=1.0 < 1.5 + 0.18
    coll, count = checker.test_corridor_collision(0.0, rho_q, obstacles_in_path)
    print(f"Sample 2 [Obstacle in Corridor: (X=0.05, Z=-1.0)]: Collision = {coll} (count={count})")
    assert coll and count == 1, "Failed to detect obstacle in corridor"

    res = checker.evaluate(0.0, -2.0, obstacles_in_path)
    assert res.status == "reselect" and res.cause == "obstacle", f"Expected reselect, got {res.status}"
    print(f"  -> Safety Result: status='{res.status}', cause='{res.cause}'")

    # Test 3: Narrow Doorway / Obstacle beside the corridor (|h| = 0.25m > r_a = 0.18m)
    doorframe_obstacles = np.array([[0.25, -1.0], [-0.25, -1.0]])  # Doorway of width 0.50m
    coll, count = checker.test_corridor_collision(0.0, rho_q, doorframe_obstacles)
    print(f"Sample 3 [Clear Doorway: Door posts at X=±0.25m (|h|=0.25m > r_a=0.18m)]:")
    print(f"  -> Corridor Check: Collision = {coll} (count={count}) -> PATH IS CLEAR!")
    assert not coll, "Corridor incorrectly blocked by clear doorway"

    res_door = checker.evaluate(0.0, -2.0, doorframe_obstacles)
    assert res_door.status == "executed", f"Expected executed for clear door, got {res_door.status}"
    print(f"  -> Safety Result: status='{res_door.status}', turn_steps={res_door.turn_steps}, forward_steps={res_door.forward_steps}")
    print("[PASS] Module 2 verified cleanly.")


def test_section_3_occupancy_grid_and_bev():
    print("\n" + "=" * 70)
    print("MODULE 3: ALLOCENTRIC BEV OCCUPANCY GRID & RENDERING (mapping)")
    print("=" * 70)

    grid = OccupancyGrid2D(grid_size=256, resolution_m=0.15)
    renderer = BEVRenderer(mode="full")

    print(f"Occupancy Grid: {grid.grid_size}x{grid.grid_size} px @ {grid.resolution:.2f} m/px ({grid.grid_size * grid.resolution:.1f}m span)")

    # Update with some obstacles at (0, -2) and free space ray
    obstacle_points = np.array([[0.0, -2.0], [0.5, -2.0], [-0.5, -2.0]])
    free_endpoints = np.array([[0.0, -4.0], [1.0, -4.0], [-1.0, -4.0]])

    grid.update(agent_x_m=0.0, agent_z_m=0.0, obstacle_points_world_xz=obstacle_points, free_endpoints_world_xz=free_endpoints)
    uint_map = grid.to_uint8_image()
    free_cells = int(np.sum(uint_map == 255))
    occ_cells = int(np.sum(uint_map == 0))
    print(f"Grid Updated: free_cells={free_cells}, occ_cells={occ_cells}")
    assert free_cells > 0 and occ_cells > 0, "Grid cells not updated"

    # Render BEV Image
    bev_img = renderer.render(
        grid=grid,
        node_poses=[(0.0, 0.0, 0.0)],
        current_node_idx=1,
    )
    print(f"Rendered BEV Image: shape={bev_img.shape}, dtype={bev_img.dtype}")
    assert bev_img.shape == (256, 256, 3)
    assert bev_img.dtype == np.uint8
    print("[PASS] Module 3 verified cleanly.")


def test_section_4_memory_and_image_cache():
    print("\n" + "=" * 70)
    print("MODULE 4: EPISODIC MEMORY & ACTION HISTORY (memory)")
    print("=" * 70)

    # Test Image Cache
    cache = EpisodicImageCache()
    dummy_views = {v: np.zeros((60, 80, 3), dtype=np.uint8) for v in range(4)}
    cache.store_node_observations(node=1, rgb_views=dummy_views)
    cache.store_node_observations(node=2, rgb_views=dummy_views)
    cache.set_current_node(node=3)

    print(f"Stored nodes in cache: {len(cache._cache)} nodes")
    assert (1 in cache._cache) and (2 in cache._cache) and (3 not in cache._cache)

    # Causal check: at current step t=3, can recall node 1 and 2, but NOT 3 or 4
    res = cache.recall(node=1, view=0)
    assert res.success and res.error_message is None, f"Failed valid recall: {res.error_message}"
    print(f"Sample 1 [Valid Recall: node=1, view=0, current_node=3]: SUCCESS -> Frame returned")

    res_future = cache.recall(node=3, view=0)
    assert not res_future.success and "only allowed for past" in res_future.error_message, f"Causal check failed: {res_future.error_message}"
    print(f"Sample 2 [Causal Violation: node=3, current_node=3]: REJECTED -> '{res_future.error_message}'")

    # Test Action History
    history = ActionHistory(max_history=6)
    for i in range(1, 9):
        history.append(
            step_idx=i,
            from_node=i,
            to_node=i+1,
            action_type="move_to",
            action_params={"view": 0, "u": 0.5, "v": 0.5},
            reason=f"Step forward to landmark {i}",
            outcome={"status": "executed", "moved_m": 1.0, "turned_deg": 0.0, "collided": False}
        )
    formatted = history.format_for_prompt()
    print(f"Action History (added 8 records, window=6): Total stored = {len(history)}")
    assert len(history) == 6, f"Expected sliding window of 6, got {len(history)}"
    assert "Node 3 -> Node 4" in formatted
    assert "Node 1 -> Node 2" not in formatted  # Pruned
    print(f"History Prompt Snippet:\n{formatted[:220]}...")
    print("[PASS] Module 4 verified cleanly.")


def test_section_5_tool_dispatcher_and_multi_turn_loop():
    print("\n" + "=" * 70)
    print("MODULE 5: ACTIVE TOOL DISPATCHER & MULTI-TURN DECISION LOOP (harness)")
    print("=" * 70)

    cam_cfg = CameraConfig()
    corridor_checker = GroundCorridorChecker(robot_radius=0.18, base_margin=0.30)
    cache = EpisodicImageCache()
    executor = ToolExecutor(cam_cfg, corridor_checker, cache)

    # Create dummy depth views
    depth_views = {v: np.full((cam_cfg.image_height, cam_cfg.image_width), 2.5, dtype=np.float32) for v in range(4)}
    # Put a close obstacle in View 0 center (0.8m depth)
    depth_views[0][int(cam_cfg.image_height*0.5)-10:int(cam_cfg.image_height*0.5)+10, int(cam_cfg.image_width*0.5)-10:int(cam_cfg.image_width*0.5)+10] = 0.8

    # Query Depth
    q_res = executor.execute_query_depth(points=[{"view": 0, "u": 0.5, "v": 0.5}, {"view": 1, "u": 0.5, "v": 0.5}], depth_views=depth_views)
    print("Sample 1 [Tool query_depth]:")
    print(f"  -> View 0 (obstacle): depth_m={q_res['query_results'][0]['depth_m']:.2f}m, min_m={q_res['query_results'][0]['min_m']:.2f}m")
    print(f"  -> View 1 (clear)   : depth_m={q_res['query_results'][1]['depth_m']:.2f}m, min_m={q_res['query_results'][1]['min_m']:.2f}m")
    assert q_res['query_results'][0]['min_m'] == 0.8
    assert q_res['query_results'][1]['depth_m'] == 2.5

    # Move To on Obstacle
    obstacles_in_front = np.array([[0.0, -0.4]])  # Obstacle at 0.4m ahead in corridor
    mv_res = executor.execute_move_to(
        view=0, u=0.5, v=0.5, reason="Head forward",
        depth_views=depth_views, obstacle_points_agent_xz=obstacles_in_front
    )
    print(f"Sample 2 [Tool move_to towards obstacle]: status='{mv_res['status']}', cause='{mv_res.get('cause')}'")
    assert mv_res['status'] == "reselect" and mv_res.get('cause') == "obstacle"

    # Move To on Clear View 1
    empty_obstacles = np.empty((0, 2), dtype=np.float32)
    mv_clear = executor.execute_move_to(
        view=1, u=0.5, v=0.5, reason="Turn right into hallway",
        depth_views=depth_views, obstacle_points_agent_xz=empty_obstacles
    )
    print(f"Sample 3 [Tool move_to towards clear hallway]: status='{mv_clear['status']}', turn_steps={mv_clear.get('turn_steps')}, forward_steps={mv_clear.get('forward_steps')}")
    assert mv_clear['status'] == "executed"
    assert mv_clear.get('forward_steps', 0) > 0

    print("[PASS] Module 5 verified cleanly.")


def test_section_6_closed_loop_runner_and_metrics():
    print("\n" + "=" * 70)
    print("MODULE 6: END-TO-END CLOSED-LOOP RUNNER & METRICS EVALUATION")
    print("=" * 70)

    tmp_dir = Path("scratch/test_traces")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    logger = TraceLogger(tmp_dir)

    config = ReplicaConfig()
    embodiment = MockGroundEmbodiment()

    # Test mock VLM that queries depth, then moves forward, then calls stop
    turn_counter = 0
    def mock_vlm_policy(messages, tools):
        nonlocal turn_counter
        turn_counter += 1

        # Firewall check: messages must NEVER contain goal or ground-truth coordinates
        prompt_str = str(messages)
        assert "goal_x" not in prompt_str
        assert "goal_z" not in prompt_str
        assert "PathFinder" not in prompt_str

        if turn_counter == 1:
            return {
                "content": "",
                "tool_calls": [{"name": "query_depth", "arguments": {"points": [{"view": 0, "u": 0.5, "v": 0.5}]}}],
            }
        elif turn_counter == 2:
            return {
                "content": "",
                "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Moving toward counter"}}],
            }
        else:
            return {
                "content": "",
                "tool_calls": [{"name": "stop", "arguments": {"reason": "Arrived at destination"}}],
            }

    episode_data = {
        "episode_id": "ep_test_001",
        "scene_id": "mp3d_test_scene",
        "instruction": "Walk straight ahead down the corridor and stop at the counter.",
        "goal_x": 0.0,
        "goal_z": -2.0,  # Goal is 2.0m ahead
    }

    res: EpisodeRunResult = run_episode(
        episode_data=episode_data,
        embodiment=embodiment,
        config=config,
        vlm_client_fn=mock_vlm_policy,
        trace_logger=logger,
    )

    print("Closed-Loop Episode Execution Summary:")
    print(f"  Episode ID:      {res.episode_id}")
    print(f"  Decision Nodes:  {res.total_steps}")
    print(f"  Trajectory Len:  {res.metrics.trajectory_length_m:.2f}m")
    print(f"  Final Error NE:  {res.metrics.navigation_error_m:.2f}m")
    print(f"  Success (SR):    {res.metrics.success} (SR=True requires stop=True and NE <= 3.0m)")
    print(f"  Oracle (OSR):    {res.metrics.oracle_success}")
    print(f"  Path Weight SPL: {res.metrics.spl:.4f}")
    print(f"  Failure Tier:    {res.failure_category}")

    assert res.metrics.success is True
    assert res.metrics.oracle_success is True
    assert res.metrics.spl > 0.0
    assert res.failure_category is None

    # Test Failure Classifier on premature stop
    outcome_prem = classify_failure(
        success=False,
        oracle_success=False,
        called_stop=True,
        final_distance_to_goal=5.0,
        action_budget_exceeded=False,
        deadlock_terminated=False,
    )
    print(f"Failure Classifier Premature Stop Check: {outcome_prem}")
    assert outcome_prem == "stop_premature"

    # Test Bootstrap CI
    sample_sr = np.array([1] * 76 + [0] * 24)  # 76% SR
    boot_res = compute_bootstrap_ci(sample_sr, n_resamples=1000)
    print(f"Bootstrap 95% CI Check (N=100, 76% SR): mean={boot_res.mean:.2f}, CI=[{boot_res.ci_lower:.2f}, {boot_res.ci_upper:.2f}]")
    assert 0.65 < boot_res.ci_lower < 0.76
    assert 0.76 < boot_res.ci_upper < 0.85

    print("[PASS] Module 6 verified cleanly.")


def main():
    print("\n" + "#" * 70)
    print("STARTING FULL END-TO-END SYSTEM ROBUSTNESS CHECK")
    print("#" * 70)

    test_section_1_geometry_and_pinhole()
    test_section_2_safety_corridor_and_kinematics()
    test_section_3_occupancy_grid_and_bev()
    test_section_4_memory_and_image_cache()
    test_section_5_tool_dispatcher_and_multi_turn_loop()
    test_section_6_closed_loop_runner_and_metrics()

    print("\n" + "#" * 70)
    print("ALL 6 MODULES VERIFIED: COMPLETE END-TO-END ROBUSTNESS CONFIRMED!")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
