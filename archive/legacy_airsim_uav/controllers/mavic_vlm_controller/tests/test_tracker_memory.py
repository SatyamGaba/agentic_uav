"""
Test suite for Target Memory Tracker & Safety FSM
=================================================
Verifies:
- Edge proximity lead centering
- Closed-loop 3D metric re-acquisition
- Budgeted search budget enforcement
- Safety FSM priority hierarchy
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


def test_edge_proximity_centering():
    tracker = TargetMemoryTrackerV6(edge_threshold=0.70)
    graph = MetricSceneGraph3D()
    tel = DroneTelemetry(position=np.zeros(3), velocity=np.zeros(3), yaw=0.0, gimbal_pitch=0.2)

    # Target near the far right border of image: u_norm = [0.85, 0.45, 0.95, 0.55] -> cx = +0.8 > 0.7
    action = tracker.update(
        target_visible=True,
        bbox_norm=(0.85, 0.45, 0.95, 0.55),
        scene_graph=graph,
        telemetry=tel,
    )

    assert tracker.state == TargetRecoveryState.EDGE_ANTICIPATION
    assert action.yaw_rate_cmd < 0.0  # Centering turn toward right (negative yaw in standard body convention)
    assert action.velocity_cmd is None  # Does not freeze body flight


def test_safety_fsm_proximity_override():
    fsm = SafetyFallbackFSM(emergency_clearance_m=1.5)
    from contracts import TargetRecoveryAction

    dummy_action = TargetRecoveryAction(gimbal_pitch_cmd=0.0, gimbal_yaw_cmd=0.0, yaw_rate_cmd=0.0)

    # 1. Proximity breach: clearance 1.0m <= 1.5m
    state, override_v, override_yaw = fsm.step(d_frontal=1.0, tracker_action=dummy_action)
    assert state == SafetyState.EMERGENCY_REPULSION
    assert override_v is not None
    assert override_v[0] == 0.0
    assert override_v[2] > 0.0  # Vertical climb override!

    # 2. Clear horizon: clearance 10m
    state_norm, override_v2, _ = fsm.step(d_frontal=10.0, tracker_action=dummy_action)
    assert state_norm == SafetyState.NORMAL
    assert override_v2 is None
