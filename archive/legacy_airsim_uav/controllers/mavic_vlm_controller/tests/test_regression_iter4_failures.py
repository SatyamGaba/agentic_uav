"""
Regression Test Suite: Iteration 4 Failure Modes vs Iteration 6
==============================================================
Verifies Phase 5 exit criteria:
- Replaying Iteration-4 coasting trace proves ZERO translation lock
- Target loss triggers 3D Scene Graph guided re-acquisition instead of blind in-place rotation
- Bounded search budget prevents infinite search locking
"""

import math
import numpy as np
import pytest

from contracts import (
    DroneTelemetry,
    SceneNode,
    TargetRecoveryState,
    SafetyState,
)
from core.scene_graph_3d import MetricSceneGraph3D
from core.tracker_memory import TargetMemoryTrackerV6
from core.safety_fsm import SafetyFallbackFSM


def test_regression_no_translation_lock_on_lost_target():
    """Direct regression test of Iteration 4's coasting mode bug.
    
    In Iteration 4: target lost -> coasting mode active -> body translations LOCKED to 0.0m/s.
    In Iteration 6: target lost -> 3D scene graph reacquisition -> active forward velocity toward target!
    """
    graph = MetricSceneGraph3D()
    # Confirmed target node at World (15.0m, 20.0m, 5.0m)
    target_node = SceneNode(
        node_id=1,
        label="target_car",
        centroid_world=np.array([15.0, 20.0, 5.0]),
        dimensions_3d=np.array([4.5, 2.0, 1.5]),
        confidence=0.92,
        observation_count=3,
        last_observed_timestamp=5.0,
        first_viewpoint=np.array([0.0, 0.0, 5.0]),
        is_confirmed=True,
    )
    graph.nodes[1] = target_node

    tracker = TargetMemoryTrackerV6(reacquisition_speed=2.5)
    tracker.set_active_target(node_id=1, scene_graph=graph)

    safety_fsm = SafetyFallbackFSM(emergency_clearance_m=1.5)

    # UAV is currently at (5.0, 10.0, 5.0). Target at (15.0, 20.0, 5.0) is 14.14m away.
    telemetry = DroneTelemetry(
        position=np.array([5.0, 10.0, 5.0]),
        velocity=np.zeros(3),
        yaw=math.radians(45.0),
        timestamp=6.0,
    )

    # Iteration 4 Event: Target suddenly becomes occluded / not visible
    tracker_action = tracker.update(
        target_visible=False,
        bbox_norm=None,
        scene_graph=graph,
        telemetry=telemetry,
    )

    # 1. Verify Tracker State
    assert tracker.state == TargetRecoveryState.METRIC_3D_REACQUISITION
    assert tracker_action.velocity_cmd is not None

    # 2. THE REGRESSION ASSERTION: Velocity must NOT be zero (NO translation lock!)
    v_norm = float(np.linalg.norm(tracker_action.velocity_cmd))
    assert v_norm == pytest.approx(2.5, abs=0.1)  # Actively flying at 2.5 m/s toward target!

    # 3. Verify flight direction is pointed straight at target:
    expected_dir = (target_node.centroid_world - telemetry.position) / np.linalg.norm(target_node.centroid_world - telemetry.position)
    actual_dir = tracker_action.velocity_cmd / v_norm
    assert np.allclose(actual_dir, expected_dir, atol=1e-3)

    # 4. Verify Safety FSM approves the re-acquisition velocity without locking
    state, override_v, _ = safety_fsm.step(d_frontal=20.0, tracker_action=tracker_action)
    assert state == SafetyState.REACQUISITION
    assert override_v is not None
    assert np.linalg.norm(override_v) > 2.0


def test_regression_search_budget_enforcement():
    """Verifies that if target remains un-acquired at destination, search budget limits turns and replans."""
    graph = MetricSceneGraph3D()
    target_node = SceneNode(
        node_id=1,
        label="target_car",
        centroid_world=np.array([10.0, 10.0, 5.0]),
        dimensions_3d=np.array([4.5, 2.0, 1.5]),
        confidence=0.90,
        observation_count=3,
        last_observed_timestamp=5.0,
        first_viewpoint=np.array([0.0, 0.0, 5.0]),
        is_confirmed=True,
    )
    graph.nodes[1] = target_node

    tracker = TargetMemoryTrackerV6(search_step_budget=3)
    tracker.set_active_target(1, graph)

    # UAV has arrived within 1.0m of target coordinates: pos=(10.0, 10.5, 5.0)
    telemetry = DroneTelemetry(
        position=np.array([10.0, 10.5, 5.0]),
        velocity=np.zeros(3),
        yaw=0.0,
        timestamp=10.0,
    )

    # Step 1: Target still missing -> Budgeted search step 1
    act1 = tracker.update(False, None, graph, telemetry)
    assert tracker.state == TargetRecoveryState.BUDGETED_SEARCH
    assert act1.trigger_replan is False

    # Step 2: Budgeted search step 2
    act2 = tracker.update(False, None, graph, telemetry)
    assert tracker.state == TargetRecoveryState.BUDGETED_SEARCH
    assert act2.trigger_replan is False

    # Step 3: Budgeted search step 3 (last step of budget=3)
    act3 = tracker.update(False, None, graph, telemetry)
    assert tracker.state == TargetRecoveryState.BUDGETED_SEARCH

    # Step 4: Budget exhausted -> Declares lost and triggers replan!
    act4 = tracker.update(False, None, graph, telemetry)
    assert tracker.state == TargetRecoveryState.LOST_DECLARED
    assert act4.trigger_replan is True
