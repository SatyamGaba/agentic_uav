"""
Test suite for 3D Metric Scene Graph
====================================
Verifies Phase 2 exit criteria:
- Direct depth backprojection recovers known synthetic 3D coordinates
- Deduplication fuses multi-view observations within dynamic radius
- Single-view transient detections remain unconfirmed (anti-hallucination)
- Physical dimensions are measured directly from point distributions without dictionaries
"""

import math
import numpy as np
import pytest

from contracts import DroneTelemetry
from core.scene_graph_3d import (
    compute_camera_to_world_matrix,
    backproject_bbox,
    robust_centroid_and_extent,
    MetricSceneGraph3D,
)


def test_synthetic_backprojection_accuracy():
    """Validates that a synthetic obstacle with known depth is backprojected to exact world coordinates."""
    # UAV at world origin (0, 0, 10m), heading East (yaw=0), camera level (pitch=0)
    telemetry = DroneTelemetry(
        position=np.array([0.0, 0.0, 10.0]),
        velocity=np.zeros(3),
        yaw=0.0, # Facing East (+X)
        gimbal_pitch=0.0,
    )
    T_wc = compute_camera_to_world_matrix(telemetry)

    # 400x300 camera with fx=fy=200, cx=200, cy=150
    K = np.array([
        [200.0, 0.0, 200.0],
        [0.0, 200.0, 150.0],
        [0.0, 0.0, 1.0],
    ])

    # Create synthetic depth map: flat plane at depth = 20.0m
    depth_map = np.full((300, 400), 20.0, dtype=np.float32)

    # Center bounding box [0.4, 0.4, 0.6, 0.6] centered at (cx=200, cy=150)
    bbox_norm = (0.4, 0.4, 0.6, 0.6)
    pts_w = backproject_bbox(bbox_norm, depth_map, K, T_wc)

    assert len(pts_w) > 50
    centroid, extents, yaw_box = robust_centroid_and_extent(pts_w)

    # Since UAV is at (0, 0, 10) facing East (+X) and target is at depth 20m straight ahead:
    # Expected world centroid: X=20.0m, Y=0.0m, Z=10.0m
    assert centroid[0] == pytest.approx(20.0, abs=0.2)
    assert centroid[1] == pytest.approx(0.0, abs=0.2)
    assert centroid[2] == pytest.approx(10.0, abs=0.2)


def test_dedup_and_anti_hallucination_confirmation():
    """Verifies that 1 observation is an unconfirmed candidate, and 2 multi-view observations confirm it."""
    graph = MetricSceneGraph3D(match_radius_base=2.5, min_views_to_confirm=2, min_viewpoint_baseline_m=1.5)
    K = np.array([[200.0, 0, 200.0], [0, 200.0, 150.0], [0, 0, 1]])

    # Viewpoint 1: UAV at (0, 0, 8m), facing East (+X)
    tel_1 = DroneTelemetry(position=np.array([0.0, 0.0, 8.0]), velocity=np.zeros(3), yaw=0.0)
    T_wc_1 = compute_camera_to_world_matrix(tel_1)
    depth_1 = np.full((300, 400), 15.0, dtype=np.float32)

    node_1 = graph.add_or_update_detection(
        label="manor_house",
        bbox_norm=(0.45, 0.45, 0.55, 0.55), # centered at (200, 150) -> world (15.0, 0.0, 8.0)
        depth_map=depth_1,
        K=K,
        T_wc=T_wc_1,
        uav_position=tel_1.position,
        timestamp=1.0,
    )
    assert node_1 is not None
    assert node_1.node_id == 1
    assert node_1.is_confirmed is False  # Anti-hallucination gate: not confirmed on 1st view!
    assert len(graph.get_confirmed_nodes()) == 0

    # Viewpoint 2: UAV moved laterally to (0.0, 1.5, 8m) (baseline = 1.5m)
    # The landmark at (15.0, 0.0, 8.0) now has relative Y = -1.5m
    # In camera frame: X_c = +1.5m -> pixel u = 200 + 200*(1.5/15) = 220 -> norm_u = 220/400 = 0.55
    tel_2 = DroneTelemetry(position=np.array([0.0, 1.5, 8.0]), velocity=np.zeros(3), yaw=0.0)
    T_wc_2 = compute_camera_to_world_matrix(tel_2)
    depth_2 = np.full((300, 400), 15.0, dtype=np.float32)

    node_2 = graph.add_or_update_detection(
        label="manor_house",
        bbox_norm=(0.50, 0.45, 0.60, 0.55), # centered at u=0.55 (220px)
        depth_map=depth_2,
        K=K,
        T_wc=T_wc_2,
        uav_position=tel_2.position,
        timestamp=3.0,
    )

    # Must be fused into existing node 1, NOT duplicate
    assert node_2.node_id == 1
    assert node_2.observation_count == 2
    assert node_2.is_confirmed is True  # Now confirmed!
    assert len(graph.get_confirmed_nodes()) == 1


def test_zero_volume_dictionary_dimension_recovery():
    """Verifies that physical building extent is measured from depth spread, not fixed tables."""
    np.random.seed(42)
    # Synthetic point cloud spanning X: [10 to 22] (12m), Y: [-4 to 4] (8m), Z: [0 to 14] (14m)
    x = np.random.uniform(10.0, 22.0, 2000)
    y = np.random.uniform(-4.0, 4.0, 2000)
    z = np.random.uniform(0.0, 14.0, 2000)
    pts = np.column_stack([x, y, z])

    centroid, dims, yaw_box = robust_centroid_and_extent(pts)

    assert centroid[0] == pytest.approx(16.0, abs=0.8)
    assert centroid[1] == pytest.approx(0.0, abs=0.8)
    assert centroid[2] == pytest.approx(7.0, abs=0.8)

    # Measured spans should approximate the distribution span without any prior dictionary!
    assert dims[0] == pytest.approx(10.8, abs=1.5)  # 90% quantile of 12m
    assert dims[1] == pytest.approx(7.2, abs=1.5)   # 90% quantile of 8m
    assert dims[2] == pytest.approx(12.6, abs=1.5)  # 90% quantile of 14m
