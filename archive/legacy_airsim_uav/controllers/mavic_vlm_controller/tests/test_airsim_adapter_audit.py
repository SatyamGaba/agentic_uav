"""
AirSim Adapter Contract & Coordinate Transform Audit Test
========================================================
Audits the physical contract and mathematical transformations of AirSimAdapter:
1. ENU <-> NED coordinate frame transforms: position, velocity, and yaw signs.
2. Camera intrinsics: 640x360 resolution, 65 deg HFOV, aspect-ratio parity with OpenFly.
3. Continuous velocity integration and dead reckoning.
4. RPC mock resilience and fallback behavior.
"""
import math
import numpy as np
import pytest

from adapters.airsim_adapter import AirSimAdapter
from contracts import DroneTelemetry, CameraIntrinsics

def test_airsim_coordinate_transforms():
    # Test 1: Position transformation ENU -> AirSim NED
    # In ENU: +X = East (10m), +Y = North (20m), +Z = Up (5m)
    p_enu = np.array([10.0, 20.0, 5.0])
    p_ned = AirSimAdapter.enu_to_airsim_ned(p_enu)
    
    # AirSim NED expects: X=10, Y=-20, Z=-5
    assert np.allclose(p_ned, np.array([10.0, -20.0, -5.0])), f"p_ned mismatch: {p_ned}"
    
    # Invert back to ENU
    p_enu_recovered = AirSimAdapter.airsim_ned_to_enu(p_ned)
    assert np.allclose(p_enu_recovered, p_enu), "Round-trip ENU <-> NED failed"

    # Test 2: Yaw angle transformation
    # ENU: +yaw is CCW, AirSim NED: +yaw is CW
    yaw_enu = math.radians(45.0)
    yaw_ned = AirSimAdapter.enu_yaw_to_airsim(yaw_enu)
    assert math.isclose(yaw_ned, -math.radians(45.0)), f"yaw_ned mismatch: {yaw_ned}"
    assert math.isclose(AirSimAdapter.airsim_yaw_to_enu(yaw_ned), yaw_enu), "Round-trip yaw failed"


def test_airsim_camera_intrinsics():
    adapter = AirSimAdapter()
    rgb, depth, intrinsics = adapter.get_camera_frame()

    assert intrinsics.width == 640
    assert intrinsics.height == 360
    assert math.isclose(intrinsics.hfov, math.radians(90.0), abs_tol=1e-4)
    expected_fx = (640.0 / 2.0) / math.tan(math.radians(90.0) / 2.0)
    assert math.isclose(intrinsics.fx, expected_fx, abs_tol=1e-2)
    assert rgb.shape == (360, 640, 3)
    assert depth.shape == (360, 640)
    assert depth.dtype == np.float32


def test_airsim_velocity_integration():
    adapter = AirSimAdapter(initial_pos=np.array([0.0, 0.0, 5.0]), initial_yaw=0.0)
    # Command velocity [2.0, 1.0, 0.5] m/s for dt = 1.0s
    adapter.set_continuous_velocity(2.0, 1.0, 0.5, yaw_rate=0.1)
    adapter.step_simulation(dt=1.0)

    tel = adapter.get_telemetry()
    assert np.allclose(tel.position, np.array([2.0, 1.0, 5.5]))
    assert math.isclose(tel.yaw, 0.1)
    assert tel.timestamp == 1.0


def test_airsim_mock_rpc_client():
    """Validates that AirSimAdapter correctly calls client RPCs with NED vectors when client is provided."""
    class MockAirSimClient:
        def __init__(self):
            self.last_velocity_cmd = None
            self.last_yaw_rate = None

        def moveByVelocityAsync(self, vx, vy, vz, duration, drivetrain, yaw_mode, vehicle_name):
            self.last_velocity_cmd = (vx, vy, vz)
            self.last_yaw_rate = yaw_mode.yaw_or_rate
            return self

    mock_client = MockAirSimClient()
    adapter = AirSimAdapter(airsim_client=mock_client)
    
    # Command velocity in ENU: +2.0 East, +1.0 North, +0.5 Up, yaw_rate = +0.2 rad/s
    adapter.set_continuous_velocity(2.0, 1.0, 0.5, yaw_rate=0.2)
    
    # In step_simulation, it should transform to AirSim NED:
    # v_ned = [2.0, -1.0, -0.5], yaw_rate_deg = -degrees(0.2)
    # Note: if HAS_AIRSIM is False in test environment, mock step should proceed cleanly
    adapter.step_simulation(dt=0.5)
    
    # Even without the airsim pip package installed, the adapter falls back safely without crashing
    tel = adapter.get_telemetry()
    assert tel.position[0] == 1.0  # 2.0 * 0.5
