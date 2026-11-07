"""
Base Continuous UAV Adapter Interface
=====================================
Formal abstract interface for all simulated and physical UAV platforms.
"""

from __future__ import annotations
import abc
from typing import Tuple
import numpy as np

from contracts import DroneTelemetry, CameraIntrinsics


class ContinuousUAVController(abc.ABC):
    """Abstract interface defining 6-DoF continuous flight dynamics and perception."""

    @abc.abstractmethod
    def set_continuous_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        """Commands continuous 3D velocity in World ENU frame and yaw rate in rad/s."""
        pass

    @abc.abstractmethod
    def set_gimbal_angles(self, pitch: float, yaw: float = 0.0) -> None:
        """Commands 2-axis gimbal attitude in radians."""
        pass

    @abc.abstractmethod
    def get_telemetry(self) -> DroneTelemetry:
        """Retrieves real-time 6-DoF pose, velocity, and gimbal state."""
        pass

    @abc.abstractmethod
    def get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """Retrieves synchronized (rgb_image, depth_map, intrinsics)."""
        pass

    @abc.abstractmethod
    def step_simulation(self, dt: float) -> None:
        """Advances simulator physics by dt seconds."""
        pass
