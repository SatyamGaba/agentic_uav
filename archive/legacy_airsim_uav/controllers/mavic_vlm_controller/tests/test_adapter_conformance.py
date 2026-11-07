"""
Test suite for Adapter Conformance
==================================
Verifies Phase 6 exit criteria:
- OpenFlyBenchmarkAdapter, WebotsAdapter, and AirSimAdapter all strictly conform
  to ContinuousUAVController interface contracts.
"""

import math
import numpy as np
import pytest

from contracts import DroneTelemetry, CameraIntrinsics
from adapters.base_adapter import ContinuousUAVController
from adapters.openfly_adapter import OpenFlyBenchmarkAdapter, ACTION_FORWARD_3M, ACTION_TURN_LEFT_30, ACTION_TURN_RIGHT_30
from adapters.webots_adapter import WebotsAdapter
from adapters.airsim_adapter import AirSimAdapter


@pytest.mark.parametrize("adapter_cls", [OpenFlyBenchmarkAdapter, WebotsAdapter, AirSimAdapter])
def test_adapter_interface_conformance(adapter_cls):
    """Verifies that all adapters inherit from ContinuousUAVController and implement all methods."""
    adapter = adapter_cls()
    assert isinstance(adapter, ContinuousUAVController)

    # 1. Command velocity
    adapter.set_continuous_velocity(2.0, 0.0, 0.5, yaw_rate=0.1)

    # 2. Command gimbal
    adapter.set_gimbal_angles(pitch=0.35, yaw=0.0)

    # 3. Retrieve telemetry
    tel = adapter.get_telemetry()
    assert isinstance(tel, DroneTelemetry)
    assert tel.altitude > 0.0

    # 4. Retrieve camera frame
    rgb, depth, intrinsics = adapter.get_camera_frame()
    assert isinstance(rgb, np.ndarray)
    assert isinstance(depth, np.ndarray)
    assert isinstance(intrinsics, CameraIntrinsics)
    assert rgb.shape[0] == depth.shape[0]
    assert rgb.shape[1] == depth.shape[1]

    # 5. Step simulation
    prev_time = tel.timestamp
    adapter.step_simulation(dt=0.1)
    new_tel = adapter.get_telemetry()
    assert new_tel.timestamp == pytest.approx(prev_time + 0.1, abs=1e-4)


def test_openfly_action_quantization():
    """Verifies that continuous velocity and yaw rates are cleanly mapped to discrete OpenFly actions."""
    adapter = OpenFlyBenchmarkAdapter()

    # Large forward velocity -> ACTION_FORWARD_3M (0)
    act_fwd = adapter.quantize_action(v_cmd=np.array([2.5, 0.0, 0.0]), yaw_rate_cmd=0.0, dt=1.0)
    assert act_fwd == ACTION_FORWARD_3M

    # Positive yaw rate -> ACTION_TURN_LEFT_30 (code 2, CCW)
    act_turn = adapter.quantize_action(v_cmd=np.array([2.0, 0.0, 0.0]), yaw_rate_cmd=math.radians(20.0), dt=1.0)
    assert act_turn == ACTION_TURN_LEFT_30
