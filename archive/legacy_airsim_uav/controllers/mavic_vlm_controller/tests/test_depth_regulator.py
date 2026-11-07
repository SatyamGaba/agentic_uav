"""
Test suite for Depth-Odometry In-Loop Regulator
===============================================
Verifies Phase 4 exit criteria:
- Emergency zero-speed braking when clearance <= d_margin
- Smooth progressive scaling when approaching obstacles
- Dynamic stopping distance calculation
"""

import numpy as np
import pytest
from core.depth_regulator import DepthOdometryRegulator


def test_emergency_zero_braking():
    """Crucial safety invariant: forward velocity command MUST be exactly 0.0 when clearance <= 1.5m."""
    regulator = DepthOdometryRegulator(d_margin=1.5, d_slowdown=8.0)

    # 1. Direct collision proximity: clearance = 1.2m <= 1.5m margin
    v_scaled = regulator.scale_forward_velocity(v_desired=3.0, d_frontal=1.2, current_speed=2.0)
    assert v_scaled == 0.0

    # 2. Exact boundary: clearance = 1.5m
    assert regulator.scale_forward_velocity(v_desired=3.0, d_frontal=1.5, current_speed=1.0) == 0.0

    # 3. Clear horizon: clearance = 25m > slowdown distance (8m)
    v_open = regulator.scale_forward_velocity(v_desired=3.0, d_frontal=25.0, current_speed=2.5)
    assert v_open == pytest.approx(3.0, abs=1e-3)

    # 4. In between: clearance = 4.0m -> smooth progressive reduction
    v_mid = regulator.scale_forward_velocity(v_desired=3.0, d_frontal=4.0, current_speed=1.5)
    assert 0.0 < v_mid < 3.0


def test_zone_depth_extraction():
    regulator = DepthOdometryRegulator(grid_rows=3, grid_cols=3)
    
    # 300x300 depth map where central frontal patch has an obstacle at 2.0m, everywhere else is 20m
    depth_map = np.full((300, 300), 20.0, dtype=np.float32)
    depth_map[100:200, 100:200] = 2.0  # Center sector (row=1, col=1)

    zones = regulator.compute_zone_depths(depth_map)
    assert zones.shape == (3, 3)
    assert zones[1, 1] == pytest.approx(2.0, abs=0.1)
    assert zones[0, 0] == pytest.approx(20.0, abs=0.1)

    frontal = regulator.frontal_clearance(zones)
    assert frontal == pytest.approx(2.0, abs=0.1)


def test_open_sky_sentinel_and_sparse_obstacle_regression():
    """Regression test for the Sky-as-Obstacle defect found in Episode 200 live run.
    
    1. A patch filled with AirSim sky sentinels (26544m and +inf) must report max_clearance (100.0m), NOT 0.0m.
    2. A patch with sparse returns (5 pixels at 4.0m representing a wire/pole) must report 4.0m (not 0.0m or 100m).
    3. A patch with true near-contact occlusion (pixels <= 0.1m) must report 0.0m.
    4. An all-zero sensor blackout frame must report 0.0m.
    """
    regulator = DepthOdometryRegulator(grid_rows=3, grid_cols=3)

    # 1. Sky-dominated depth map: Upper two rows are sky (26544.0m and inf), bottom row is ground (15.0m)
    depth_sky = np.full((300, 300), 26544.0, dtype=np.float32)
    depth_sky[:100, :100] = np.inf
    depth_sky[200:, :] = 15.0  # Ground at bottom

    zones_sky = regulator.compute_zone_depths(depth_sky)
    # Upper sky zones must be 100.0m, not 0.0m
    assert zones_sky[0, 0] == 100.0, f"Expected 100.0m for sky sentinel, got {zones_sky[0, 0]}"
    assert zones_sky[0, 1] == 100.0
    assert zones_sky[2, 1] == pytest.approx(15.0, abs=0.1)

    # 2. Sparse obstacle: only 5 pixels detected at 4.2m in zone (1, 1), rest is sky
    depth_sparse = np.full((300, 300), 26544.0, dtype=np.float32)
    depth_sparse[150:155, 150] = 4.2  # 5 pixels at 4.2m (thin wire/pole)
    zones_sparse = regulator.compute_zone_depths(depth_sparse)
    assert zones_sparse[1, 1] == pytest.approx(4.2, abs=0.01), f"Expected 4.2m for sparse obstacle, got {zones_sparse[1, 1]}"

    # 3. Near-contact occlusion: lens covered by hand/wall (0.05m)
    depth_contact = np.full((300, 300), 20.0, dtype=np.float32)
    depth_contact[100:200, 100:200] = 0.05
    zones_contact = regulator.compute_zone_depths(depth_contact)
    assert zones_contact[1, 1] == 0.0, f"Expected 0.0m for near-contact hazard, got {zones_contact[1, 1]}"

    # 4. Total sensor failure: all zeros
    depth_dead = np.zeros((300, 300), dtype=np.float32)
    zones_dead = regulator.compute_zone_depths(depth_dead)
    assert np.all(zones_dead == 0.0), "Expected all zeros for complete sensor blackout"


def test_pure_sky_fixture_zero_valid_returns():
    """Explicit fixture for outdoor flight: zero valid depth returns (all sentinel/sky).
    
    Verifies that facing pure open sky (common in high-altitude or upward pitch flight)
    reports full clearance (100.0m) and commands full cruise speed, NOT a zero-speed emergency brake.
    """
    regulator = DepthOdometryRegulator(d_margin=1.5, d_slowdown=8.0)
    
    # AirSim returns 26544m or inf for sky rays; ground plane is absent
    pure_sky_frame = np.full((300, 300), 26544.0, dtype=np.float32)
    pure_sky_frame[50:100, 50:100] = np.inf
    
    zones = regulator.compute_zone_depths(pure_sky_frame, max_clearance=100.0)
    assert np.all(zones == 100.0), f"All zones must report 100.0m clearance in pure sky, got: {zones}"
    
    frontal_clearance = regulator.frontal_clearance(zones)
    assert frontal_clearance == 100.0
    
    # Must NOT command zero speed
    v_cmd = regulator.scale_forward_velocity(v_desired=3.0, d_frontal=frontal_clearance, current_speed=2.0)
    assert v_cmd == pytest.approx(3.0, abs=1e-3), f"Pure sky should command full cruising speed (3.0m/s), got: {v_cmd}"


