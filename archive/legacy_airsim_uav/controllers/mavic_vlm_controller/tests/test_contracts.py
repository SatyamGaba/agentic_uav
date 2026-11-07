"""
Test suite for Iteration 6 Data Contracts & Schemas
==================================================
Verifies schema validation, serialization, coordinate conventions,
and zero circular dependencies.
"""

import math
import numpy as np
import pytest
from contracts import (
    DroneTelemetry,
    CameraIntrinsics,
    SubgoalIntent,
    Subgoal,
    SceneNode,
    MarkedCandidate,
    MarkSelection,
    TargetRecoveryState,
    TargetRecoveryAction,
    SafetyState,
)


def test_drone_telemetry_instantiation():
    pos = np.array([10.5, 20.0, 5.0], dtype=np.float64)
    vel = np.array([2.0, 0.0, -0.5], dtype=np.float64)
    telemetry = DroneTelemetry(
        position=pos,
        velocity=vel,
        yaw=math.radians(45.0),
        pitch=0.0,
        roll=0.0,
        yaw_rate=0.1,
        gimbal_pitch=0.25,
        gimbal_yaw=0.0,
        timestamp=100.5,
    )
    assert telemetry.altitude == 5.0
    assert abs(telemetry.speed - 2.0) < 1e-4
    assert telemetry.yaw > 0.0


def test_camera_intrinsics_unprojection():
    # 640x480 camera with fx=fy=500, cx=320, cy=240
    K = np.array([
        [500.0, 0.0, 320.0],
        [0.0, 500.0, 240.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    intrinsics = CameraIntrinsics(
        K=K,
        width=640,
        height=480,
        vfov=math.radians(50.0),
        hfov=math.radians(65.0),
    )
    assert intrinsics.fx == 500.0
    assert intrinsics.cx == 320.0

    # Test center pixel at 10m depth -> [0, 0, 10] in camera RDF frame
    p_center = intrinsics.unproject_pixel(320.0, 240.0, 10.0)
    assert np.allclose(p_center, [0.0, 0.0, 10.0], atol=1e-5)

    # Test offset pixel (420, 240) -> offset along +X right
    p_right = intrinsics.unproject_pixel(420.0, 240.0, 10.0)
    assert p_right[0] == pytest.approx(2.0, abs=1e-4)
    assert p_right[1] == pytest.approx(0.0, abs=1e-4)
    assert p_right[2] == pytest.approx(10.0, abs=1e-4)


def test_subgoal_pydantic_serialization():
    subgoal = Subgoal(
        subgoal_id=42,
        intent=SubgoalIntent.TARGET_APPROACH,
        waypoint_xyz=(15.0, 25.0, 8.0),
        target_node_id=3,
        v_target=2.5,
        caution_level="high",
        termination_condition="distance_to_target_lt_3m",
        issued_at=12.34,
        expiry_s=5.0,
        detector_queries=["red brick chimney", "manor house"],
    )

    json_str = subgoal.model_dump_json()
    assert "INTENT_TARGET_APPROACH" in json_str
    assert "red brick chimney" in json_str

    deserialized = Subgoal.model_validate_json(json_str)
    assert deserialized.subgoal_id == 42
    assert deserialized.intent == SubgoalIntent.TARGET_APPROACH
    assert deserialized.waypoint_xyz == (15.0, 25.0, 8.0)
    assert deserialized.target_node_id == 3
    assert deserialized.detector_queries == ["red brick chimney", "manor house"]


def test_mark_selection_schema():
    sel = MarkSelection(
        selected_mark_id=2,
        confidence=0.92,
        reasoning="Mark #2 matches the two-story manor with red brick facade.",
    )
    data = sel.model_dump()
    assert data["selected_mark_id"] == 2
    assert data["confidence"] == 0.92

    none_sel = MarkSelection(
        selected_mark_id=None,
        confidence=0.15,
        reasoning="None of the candidates match target description.",
    )
    assert none_sel.selected_mark_id is None


def test_scene_node_structure():
    node = SceneNode(
        node_id=1,
        label="solar_panel_array",
        centroid_world=np.array([25.0, 40.0, 6.2]),
        dimensions_3d=np.array([8.0, 6.0, 1.2]),
        confidence=0.88,
        observation_count=3,
        last_observed_timestamp=20.5,
        first_viewpoint=np.array([10.0, 15.0, 8.0]),
        is_confirmed=True,
        yaw_box=0.45,
    )
    assert node.is_confirmed is True
    assert node.observation_count == 3
    assert node.dimensions_3d[0] == 8.0


def test_recovery_and_safety_enums():
    assert SafetyState.NORMAL == "NORMAL"
    assert SafetyState.EMERGENCY_REPULSION == "EMERGENCY_REPULSION"
    assert TargetRecoveryState.EDGE_ANTICIPATION == "EDGE_ANTICIPATION"
    assert TargetRecoveryState.METRIC_3D_REACQUISITION == "METRIC_3D_REACQUISITION"
