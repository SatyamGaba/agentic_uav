"""
Test suite for Continuous Kinematics & 3D APF Planner
=====================================================
Verifies Phase 4 exit criteria:
- Correct repulsive force direction away from obstacles
- Altitude envelope bounding
- Jerk-limited acceleration smoothing
"""

import math
import numpy as np
import pytest
from core.kinematics import JerkLimitedIntegrator, APFPlanner


def test_apf_repulsion_direction():
    """Verifies that an obstacle on the right generates a repulsive force pushing left."""
    planner = APFPlanner(k_att=1.0, k_rep=10.0, d_0=6.0, v_max=4.0)

    # UAV at (0, 0, 5m), facing East (yaw=0). Target is straight ahead at (10, 0, 5m)
    current_pos = np.array([0.0, 0.0, 5.0])
    target_wpt = np.array([10.0, 0.0, 5.0])

    # 3x3 depth map: obstacle at 1.5m on the right (col=2), open ahead (col=1 at 20m)
    zone_depths = np.full((3, 3), 20.0, dtype=np.float32)
    zone_depths[1, 2] = 1.5  # Right sector

    force = planner.compute_force(current_pos, target_wpt, zone_depths, uav_yaw=0.0)

    # Force should maintain forward progress (+X), but push LEFT (+Y) away from right obstacle
    assert force[0] > 0.0  # Moving forward toward waypoint
    assert force[1] > 0.0  # Pushed left (+Y) away from the obstacle on the right (-Y)!


def test_apf_altitude_envelope():
    """Verifies vertical repulsion when approaching ground ceiling."""
    planner = APFPlanner(z_min=2.0, z_max=20.0)

    # UAV too low: Z = 0.5m < z_min (2.0m)
    low_pos = np.array([0.0, 0.0, 0.5])
    wpt = np.array([5.0, 0.0, 0.5])
    zones = np.full((3, 3), 20.0, dtype=np.float32)

    force = planner.compute_force(low_pos, wpt, zones, uav_yaw=0.0)
    assert force[2] > 0.0  # Upward push to restore min altitude


def test_jerk_limited_integrator():
    """Verifies that sudden step command in velocity is rate-limited by acceleration and jerk."""
    integrator = JerkLimitedIntegrator(a_max=2.0, j_max=5.0)

    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)

    # Step command: suddenly command 10 m/s forward
    v_cmd = np.array([10.0, 0.0, 0.0])
    dt = 0.05  # 20 Hz tick

    # Advance 1 tick
    new_pos, new_vel, new_acc = integrator.step(pos, vel, acc, v_cmd, dt)

    # Acceleration should not exceed jerk * dt
    assert np.linalg.norm(new_acc) <= 5.0 * dt + 1e-4
    assert new_vel[0] < 1.0  # Did not jump to 10 m/s in 50ms!
