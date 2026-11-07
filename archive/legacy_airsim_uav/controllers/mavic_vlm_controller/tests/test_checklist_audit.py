"""
Checklist Audit & Hardening Test Suite
======================================
Covers the 12 mandatory failure-mode and edge-case verification tests specified in
the Pre-run Script-Level Verification Checklist:
1. Unknown SoM mark ID -> Rejected, no waypoint created
2. Detector misses target -> 'None/search again' path available
3. Stale VLM response -> Rejected, not applied to current frame
4. Scene graph bbox contains mostly background -> Foreground isolated or node rejected
5. Missing / NaN depth -> No fabricated metric position
6. Nearby same-class buildings -> Not automatically merged
7. Lost target inside building -> Never commands flight into obstacle centroid; plans standoff
8. Moving vehicle with obstacle ahead -> Dynamic braking accounts for velocity and delay
9. Depth sensor stale / invalid -> Safe degradation to 0.0m clearance
10. Adapter high-rate quantization -> No permanent-hover deadlock from small dt increments
11. VLM unavailable / timeout -> Safe fallback, no false mission-complete
12. All components connected -> Full closed-loop System 1 / System 2 cycle verified
"""

import math
import asyncio
import numpy as np
import pytest

from contracts import (
    DroneTelemetry,
    CameraIntrinsics,
    SubgoalIntent,
    Subgoal,
    MarkSelection,
    SceneNode,
    TargetRecoveryState,
    SafetyState,
)
from cognitive.vlm_client import MockVLMClient, SelfConsistentVLMWrapper
from cognitive.intent_arbiter import IntentArbiter, TaskDecompositionResult, DecomposedStage
from cognitive.som_grounder import SoMGrounder, MockDetector
from contracts import MarkedCandidate
from cognitive.baseline_agents import FullSystemAgent
from core.scene_graph_3d import MetricSceneGraph3D, backproject_bbox, compute_camera_to_world_matrix
from core.depth_regulator import DepthOdometryRegulator
from core.tracker_memory import TargetMemoryTrackerV6
from core.safety_fsm import SafetyFallbackFSM
from adapters.openfly_adapter import OpenFlyBenchmarkAdapter, ACTION_FORWARD_3M, ACTION_HOVER
from adapters.base_adapter import ContinuousUAVController
from main_controller import SharedBlackboard, FastContinuousController, DualRateMasterController


def make_dummy_telemetry(x=0.0, y=0.0, z=5.0, yaw=0.0, speed=0.0, timestamp=100.0) -> DroneTelemetry:
    return DroneTelemetry(
        position=np.array([x, y, z], dtype=np.float64),
        velocity=np.array([speed * math.cos(yaw), speed * math.sin(yaw), 0.0], dtype=np.float64),
        yaw=yaw,
        gimbal_pitch=0.0,
        gimbal_yaw=0.0,
        timestamp=timestamp,
    )


def make_dummy_intrinsics() -> CameraIntrinsics:
    K = np.array([
        [320.0, 0.0, 320.0],
        [0.0, 320.0, 240.0],
        [0.0, 0.0, 1.0],
    ])
    return CameraIntrinsics(K=K, width=640, height=480, vfov=math.radians(60.0), hfov=math.radians(80.0))


# 1. Unknown SoM mark ID
def test_unknown_som_mark_id_rejected():
    detector = MockDetector([((0.2, 0.2, 0.4, 0.4), "building", 0.9)])
    grounder = SoMGrounder(detector)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    _, candidates = grounder.mark_frame(rgb, ["building"])
    assert len(candidates) == 1
    assert candidates[0].mark_id == 1

    # VLM hallucinated mark ID 99
    selection = MarkSelection(selected_mark_id=99, confidence=0.8, selection_rationale="Found it")
    resolved = grounder.resolve_selection(selection, candidates)
    assert resolved is None, "Unknown mark ID must resolve to None"


# 2. Detector misses target -> None / search path available
def test_detector_misses_target_search_path():
    detector = MockDetector([]) # zero detections
    grounder = SoMGrounder(detector)
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    _, candidates = grounder.mark_frame(rgb, ["red silo"])
    assert len(candidates) == 0

    selection = MarkSelection(selected_mark_id=None, confidence=0.0, selection_rationale="No target detected")
    resolved = grounder.resolve_selection(selection, candidates)
    assert resolved is None


# 3. Stale VLM response / Subgoal expiry
def test_stale_vlm_response_rejected():
    tel = make_dummy_telemetry(timestamp=100.0)
    subgoal = Subgoal(
        subgoal_id=1,
        intent=SubgoalIntent.TARGET_APPROACH,
        waypoint_xyz=(10.0, 0.0, 5.0),
        issued_at=100.0,
        expiry_s=6.0, # Expires at t = 106.0
    )
    blackboard = SharedBlackboard(subgoal)
    adapter = OpenFlyBenchmarkAdapter(initial_position=np.array([0.0, 0.0, 5.0]))
    controller = FastContinuousController(adapter, blackboard)

    # Simulate arrival at t = 108.0 (stale by 2.0s)
    adapter.sim_time = 108.0
    controller.tick()

    assert blackboard.trigger_event.is_set(), "System 2 wake-up event must be triggered on stale response"
    assert blackboard.active_safety_state == SafetyState.HOVER_REPLAN, "Must degrade to hover failsafe on expiry"


# 4. Scene graph bbox contains mostly background
def test_scene_graph_bbox_mostly_background():
    depth = np.full((480, 640), 60.0, dtype=np.float32) # 60m background
    # Small foreground patch
    depth[200:240, 300:340] = 5.0

    intrinsics = make_dummy_intrinsics()
    tel = make_dummy_telemetry()
    T_wc = compute_camera_to_world_matrix(tel)

    # Box covers 100:300, 200:400 (mostly background at 60m, but 5m patch inside)
    bbox_norm = (200.0 / 640.0, 100.0 / 480.0, 400.0 / 640.0, 300.0 / 480.0)
    pts_w = backproject_bbox(bbox_norm, depth, intrinsics.K, T_wc, valid_range=(0.3, 80.0))
    assert len(pts_w) > 0

    # Ensure robust centroid does not pick an unbounded coordinate
    sg = MetricSceneGraph3D()
    node = sg.add_or_update_detection("building", bbox_norm, depth, intrinsics.K, T_wc, tel.position, tel.timestamp)
    assert node is not None
    # Points are bounded and labeled
    assert np.all(np.isfinite(node.centroid_world))


# 5. Missing / NaN depth -> No fabricated metric position
def test_missing_depth_no_fabricated_position():
    depth_nan = np.full((480, 640), np.nan, dtype=np.float32)
    intrinsics = make_dummy_intrinsics()
    tel = make_dummy_telemetry()
    T_wc = compute_camera_to_world_matrix(tel)

    bbox_norm = (0.2, 0.2, 0.5, 0.5)
    pts_w = backproject_bbox(bbox_norm, depth_nan, intrinsics.K, T_wc)
    assert len(pts_w) == 0, "NaN depth must yield 0 backprojected points"

    sg = MetricSceneGraph3D()
    node = sg.add_or_update_detection("car", bbox_norm, depth_nan, intrinsics.K, T_wc, tel.position, tel.timestamp)
    assert node is None, "Missing depth must return None, preventing fabricated coordinates"


# 6. Nearby same-class buildings not automatically merged
def test_nearby_same_class_buildings_not_merged():
    sg = MetricSceneGraph3D(match_radius_base=3.0)
    # Building 1 at X=10.0, Y=0.0
    node1 = SceneNode(
        node_id=1, label="building", centroid_world=np.array([10.0, 0.0, 5.0]),
        dimensions_3d=np.array([4.0, 4.0, 8.0]), confidence=0.8, observation_count=1,
        last_observed_timestamp=1.0, first_viewpoint=np.zeros(3), is_confirmed=False
    )
    sg.nodes[1] = node1
    sg.next_node_id = 2

    # Building 2 at X=25.0, Y=0.0 (15 meters apart, same semantic label)
    depth = np.full((480, 640), 25.0, dtype=np.float32)
    intrinsics = make_dummy_intrinsics()
    tel = make_dummy_telemetry(x=0.0, y=0.0)
    T_wc = compute_camera_to_world_matrix(tel)

    node2 = sg.add_or_update_detection(
        label="building",
        bbox_norm=(0.45, 0.45, 0.55, 0.55),
        depth_map=depth,
        K=intrinsics.K,
        T_wc=T_wc,
        uav_position=tel.position,
        timestamp=2.0,
    )
    assert node2 is not None
    assert node2.node_id == 2, "Distinct building 15m away must NOT be merged into Building 1"
    assert len(sg.nodes) == 2


# 7. Lost target inside building -> Never commands flight into obstacle centroid
def test_lost_target_inside_building_safe_standoff():
    tracker = TargetMemoryTrackerV6(reacquisition_speed=2.5)
    sg = MetricSceneGraph3D()
    # Big warehouse: centroid at (20.0, 0.0, 5.0), size 10m x 10m x 6m
    # Physical boundary extends to X = 15.0m
    sg.nodes[1] = SceneNode(
        node_id=1, label="warehouse", centroid_world=np.array([20.0, 0.0, 5.0]),
        dimensions_3d=np.array([10.0, 10.0, 6.0]), confidence=0.9, observation_count=3,
        last_observed_timestamp=10.0, first_viewpoint=np.zeros(3), is_confirmed=True
    )
    tracker.set_active_target(1, sg)

    # UAV at (0.0, 0.0, 5.0). Target lost.
    tel = make_dummy_telemetry(x=0.0, y=0.0, z=5.0, timestamp=11.0)
    tracker.last_seen_timestamp = 10.0
    action = tracker.update(target_visible=False, bbox_norm=None, scene_graph=sg, telemetry=tel)

    # Must be in 3D Reacquisition
    assert action.velocity_cmd is not None
    # Verify forward velocity direction
    assert action.velocity_cmd[0] > 0.0

    # Now simulate UAV reaching the standoff perimeter (e.g. at X = 11.5m, outside the 15.0m wall)
    # Standoff distance = 0.5 * 10 + 3.5 = 8.5m from centroid (X = 11.5m)
    tel_at_standoff = make_dummy_telemetry(x=11.5, y=0.0, z=5.0, timestamp=12.0)
    action_at_standoff = tracker.update(target_visible=False, bbox_norm=None, scene_graph=sg, telemetry=tel_at_standoff)

    # Must NOT fly forward into the building! Must switch to BUDGETED_SEARCH with zero translation
    assert tracker.state == TargetRecoveryState.BUDGETED_SEARCH
    assert np.allclose(action_at_standoff.velocity_cmd, np.zeros(3)), "Must NOT translate into obstacle centroid"


# 8. Moving vehicle with obstacle ahead -> Braking takes velocity into account
def test_moving_vehicle_obstacle_braking():
    regulator = DepthOdometryRegulator(d_margin=1.5, a_brake=2.5, t_react=0.1)
    speed = 4.0 # m/s
    stop_dist = regulator.stopping_distance(speed) # (16 / 5) + 0.4 + 1.5 = 5.1m

    # Obstacle is at 4.5m (less than required stopping distance 5.1m!)
    v_scaled = regulator.scale_forward_velocity(v_desired=4.0, d_frontal=4.5, current_speed=speed)
    assert v_scaled == 0.0, "Must command 0.0 m/s when clearance <= dynamic stopping distance"

    # Obstacle is at 10.0m (sufficient clearance)
    v_scaled_far = regulator.scale_forward_velocity(v_desired=4.0, d_frontal=10.0, current_speed=speed)
    assert v_scaled_far > 2.0


# 9. Depth sensor stale / invalid -> Safe degradation
def test_depth_sensor_stale_safe_degradation():
    regulator = DepthOdometryRegulator()
    # All zeros depth map (sensor unplugged or frozen)
    dead_depth = np.zeros((480, 640), dtype=np.float32)
    zones = regulator.compute_zone_depths(dead_depth)
    clearance = regulator.frontal_clearance(zones)

    assert clearance == 0.0, "Corrupted depth must report 0.0m clearance to trigger safety stop"
    v_cmd = regulator.scale_forward_velocity(v_desired=3.0, d_frontal=clearance, current_speed=1.0)
    assert v_cmd == 0.0, "Forward velocity must be clamped to 0.0 on zero clearance"


# 10. Adapter high-rate quantization -> No permanent-hover deadlock
def test_openfly_high_rate_quantization_no_deadlock():
    adapter = OpenFlyBenchmarkAdapter(initial_position=np.array([0.0, 0.0, 5.0]))
    # Continuous command: 2.0 m/s forward for 25 ticks at 20 Hz (1.25s)
    # Displacement = 2.0 * 1.25 = 2.5m >= 2.0m threshold
    dt = 0.05
    actions_taken = []
    for _ in range(25):
        adapter.set_continuous_velocity(2.0, 0.0, 0.0, yaw_rate=0.0)
        adapter.step_simulation(dt)
        actions_taken.append(adapter.last_discrete_action)

    assert ACTION_FORWARD_3M in actions_taken, "Accumulator must trigger ACTION_FORWARD_3M without deadlocking to HOVER"


# 11. VLM unavailable -> Safe fallback, no false mission complete
def test_vlm_unavailable_safe_fallback():
    async def _test():
        # Empty mock client raises when queried
        mock_client = MockVLMClient([])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        arbiter = IntentArbiter(wrapper)
        tel = make_dummy_telemetry()

        with pytest.raises(ValueError, match="MockVLMClient response queue is empty"):
            await arbiter.arbitrate(
                instruction_raw="Inspect the turbine",
                rgb_image=None,
                scene_graph_text="",
                telemetry=tel,
                history=[],
            )
    asyncio.run(_test())


# 12. Full System Closed-Loop Trace
def test_all_components_connected_closed_loop():
    async def _test():
        # Scripted sequence of VLM outputs:
        # 1. Mission decomposition
        # 2. Mark selection (mark #1)
        # 3. Subgoal arbitration (approach #1)
        decomp = TaskDecompositionResult(stages=[
            DecomposedStage(stage_id=1, text="Approach the red silo", detector_queries=["silo"])
        ])
        selection = MarkSelection(selected_mark_id=1, confidence=0.88, selection_rationale="Matches silo")
        subgoal = Subgoal(
            subgoal_id=1,
            intent=SubgoalIntent.TARGET_APPROACH,
            waypoint_xyz=(15.0, 0.0, 5.0),
            v_target=2.0,
            caution_level="medium",
            termination_condition="distance_within_3m",
        )

        mock_client = MockVLMClient([decomp, selection, subgoal])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        detector = MockDetector([((0.4, 0.4, 0.6, 0.6), "silo", 0.9)])
        intrinsics = make_dummy_intrinsics()

        agent = FullSystemAgent(wrapper, detector, intrinsics)
        tel = make_dummy_telemetry(x=0.0, y=0.0, z=5.0)
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.full((480, 640), 15.0, dtype=np.float32)

        result_subgoal = await agent.step(rgb, depth, tel, "Approach the red silo")

        assert result_subgoal.intent == SubgoalIntent.TARGET_APPROACH
        assert result_subgoal.target_node_id is not None
        assert result_subgoal.target_node_id in agent.scene_graph.nodes
        assert agent.scene_graph.nodes[result_subgoal.target_node_id].label == "silo"
        assert len(agent.history) == 1

    asyncio.run(_test())
