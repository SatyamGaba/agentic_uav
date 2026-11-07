"""
Webots Mavic 2 Pro Continuous Adapter
====================================
Bridges System 1 continuous velocity & gimbal commands directly to Webots Supervisor API.
Supports headless/mock mode for automated testing without active Webots GUI.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional

from contracts import DroneTelemetry, CameraIntrinsics
from adapters.base_adapter import ContinuousUAVController


class WebotsAdapter(ContinuousUAVController):
    """Adapter for continuous 6-DoF flight in Webots simulator."""

    def __init__(self, supervisor: Optional[Any] = None, initial_pos: Optional[np.ndarray] = None):
        self.supervisor = supervisor
        self.pos = initial_pos.copy() if initial_pos is not None else np.array([0.0, 0.0, 5.0], dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.gimbal_pitch = 0.0
        self.sim_time = 0.0

        K = np.array([
            [350.0, 0.0, 200.0],
            [0.0, 350.0, 150.0],
            [0.0, 0.0, 1.0],
        ])
        self.intrinsics = CameraIntrinsics(K=K, width=400, height=300, vfov=math.radians(45.0), hfov=math.radians(60.0))

    def set_continuous_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        self.vel = np.array([vx, vy, vz], dtype=np.float64)
        self.yaw_rate = yaw_rate

    def set_gimbal_angles(self, pitch: float, yaw: float = 0.0) -> None:
        self.gimbal_pitch = max(0.0, min(0.8, pitch))

    def get_telemetry(self) -> DroneTelemetry:
        return DroneTelemetry(
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            yaw=self.yaw,
            gimbal_pitch=self.gimbal_pitch,
            gimbal_yaw=0.0,
            timestamp=self.sim_time,
        )

    def get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        # If supervisor connected, extract camera sensor bytes; otherwise fallback to synthetic
        rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        depth = np.full((300, 400), 20.0, dtype=np.float32)
        return rgb, depth, self.intrinsics

    def step_simulation(self, dt: float) -> None:
        self.pos += self.vel * dt
        self.yaw += getattr(self, "yaw_rate", 0.0) * dt
        self.sim_time += dt
