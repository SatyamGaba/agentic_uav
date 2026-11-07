"""
OpenFly Benchmark Adapter
=========================
Encapsulates OpenFly benchmark constraints (fixed forward camera, 10 discrete actions)
without compromising the general-purpose continuous core architecture.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional

from contracts import DroneTelemetry, CameraIntrinsics
from adapters.base_adapter import ContinuousUAVController


# Canonical OpenFly Action Kinematics (strictly aligned with benchmark/seen.json ground truth)
# Coordinate Convention:
# - Heading Yaw: 0 rad points along +X (East/Forward).
# - Positive Yaw (+30 deg, Action 2): Rotates CCW toward +Y (North/Left). Physically corresponds to TURN_LEFT.
# - Negative Yaw (-30 deg, Action 3): Rotates CW toward -Y (South/Right). Physically corresponds to TURN_RIGHT.
# - Lateral Translation:
#     +Y (Action 7, dy = +3.0m): Port / Left side.
#     -Y (Action 6, dy = -3.0m): Starboard / Right side.
# This convention was empirically validated against all 1,800 trajectories in seen.json.
ACTION_STOP                = 0  # dx: 0.0, dy: 0.0, dz: 0.0, yaw: 0.0 (hover / terminal arrival)
ACTION_MOVE_FORWARD_FINE   = 1  # dx: +3.0m
ACTION_TURN_LEFT_30        = 2  # yaw: +30 deg (CCW / Left) - strictly verified against seen.json
ACTION_TURN_RIGHT_30       = 3  # yaw: -30 deg (CW / Right) - strictly verified against seen.json
ACTION_CLIMB_3M            = 4  # dz: +3.0m
ACTION_DESCEND_3M          = 5  # dz: -3.0m
ACTION_STRAFE_RIGHT_3M     = 6  # dy: -3.0m (starboard / right side)
ACTION_STRAFE_LEFT_3M      = 7  # dy: +3.0m (port / left side)
ACTION_MOVE_FORWARD_MED    = 8  # dx: +6.0m
ACTION_MOVE_FORWARD_LARGE  = 9  # dx: +9.0m
ACTION_HOVER               = 0  # Station-keeping / in-flight hover maps to code 0

# Aliases for compatibility
ACTION_FORWARD_3M          = ACTION_MOVE_FORWARD_FINE
ACTION_CLIMB_2M            = ACTION_CLIMB_3M
ACTION_DESCEND_2M          = ACTION_DESCEND_3M
ACTION_LEFT_3M             = ACTION_STRAFE_LEFT_3M
ACTION_RIGHT_3M            = ACTION_STRAFE_RIGHT_3M


class OpenFlyBenchmarkAdapter(ContinuousUAVController):
    """Maps continuous 6-DoF velocity commands to canonical OpenFly discrete action tokens."""

    def __init__(self, initial_position: Optional[np.ndarray] = None, initial_yaw: float = 0.0):
        self.pos = initial_position.copy() if initial_position is not None else np.array([0.0, 0.0, 5.0], dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = initial_yaw
        self.sim_time = 0.0

        # Fixed forward-facing camera with zero gimbal tilt (strictly aligned with framework/environments/openfly_env.py)
        # 640x360 resolution, 65.0 deg horizontal FOV: focal length = (640 / 2) / tan(65 deg / 2) = 502.4px
        fx = (640.0 / 2.0) / math.tan(math.radians(65.0) / 2.0)
        fy = fx
        K = np.array([
            [fx, 0.0, 320.0],
            [0.0, fy, 180.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        vfov_rad = 2.0 * math.atan((360.0 / 2.0) / fy)
        self.intrinsics = CameraIntrinsics(K=K, width=640, height=360, vfov=vfov_rad, hfov=math.radians(65.0))
        self.last_discrete_action = ACTION_STOP
        self.yaw_rate = 0.0

        # Accumulated continuous displacement and yaw residuals across high-rate ticks
        self.accumulated_body_disp = np.zeros(3, dtype=np.float64)
        self.accumulated_yaw_rad = 0.0

    def set_continuous_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        self.vel = np.array([vx, vy, vz], dtype=np.float64)
        self.yaw_rate = yaw_rate

    def set_gimbal_angles(self, pitch: float, yaw: float = 0.0) -> None:
        # Enforce OpenFly hardware invariant: camera has zero gimbal pitch tilt
        pass

    def get_telemetry(self) -> DroneTelemetry:
        return DroneTelemetry(
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            yaw=self.yaw,
            gimbal_pitch=0.0,  # Always 0.0 in OpenFly
            gimbal_yaw=0.0,
            timestamp=self.sim_time,
        )

    def set_observation(self, rgb: np.ndarray, depth: np.ndarray, position: Optional[np.ndarray] = None, yaw: Optional[float] = None) -> None:
        """Injects live observation from benchmark environment into adapter."""
        self._current_rgb = rgb
        self._current_depth = depth
        if position is not None:
            self.pos = position.copy()
        if yaw is not None:
            self.yaw = yaw

    def get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        # Return injected observation if available, otherwise calibrated defaults
        rgb = getattr(self, "_current_rgb", None)
        if rgb is None:
            rgb = np.zeros((360, 640, 3), dtype=np.uint8)
        depth = getattr(self, "_current_depth", None)
        if depth is None:
            depth = np.full((360, 640), 25.0, dtype=np.float32)
        return rgb, depth, self.intrinsics

    def quantize_action(self, v_cmd: np.ndarray, yaw_rate_cmd: float, dt: float = 1.0) -> int:
        """Translates continuous velocity and yaw rate into OpenFly discrete action code (0-9)."""
        dyaw_deg = math.degrees(yaw_rate_cmd * dt)

        # Priority 1: Angular heading adjustment
        # Positive yaw rate -> CCW/Left (code 2), Negative yaw rate -> CW/Right (code 3)
        if dyaw_deg >= 15.0:
            return ACTION_TURN_LEFT_30
        elif dyaw_deg <= -15.0:
            return ACTION_TURN_RIGHT_30

        # Transform velocity to body frame
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        vx_w, vy_w, vz_w = v_cmd[0], v_cmd[1], v_cmd[2]
        vx_b = cos_y * vx_w + sin_y * vy_w
        vy_b = -sin_y * vx_w + cos_y * vy_w

        # Priority 2: Vertical clearance
        if vz_w * dt >= 1.5:
            return ACTION_CLIMB_3M
        elif vz_w * dt <= -1.5:
            return ACTION_DESCEND_3M

        # Priority 3: Forward progression
        if vx_b * dt >= 7.5:
            return ACTION_MOVE_FORWARD_LARGE
        elif vx_b * dt >= 4.5:
            return ACTION_MOVE_FORWARD_MED
        elif vx_b * dt >= 1.5:
            return ACTION_MOVE_FORWARD_FINE

        # Priority 4: Lateral displacement (+vy is left code 7, -vy is right code 6)
        if vy_b * dt >= 1.5:
            return ACTION_STRAFE_LEFT_3M
        elif vy_b * dt <= -1.5:
            return ACTION_STRAFE_RIGHT_3M

        return ACTION_STOP

    def step_simulation(self, dt: float) -> None:
        """Accumulates continuous velocities and executes discrete benchmark steps upon threshold crossing."""
        # Project continuous velocity into body frame
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        vx_w, vy_w, vz_w = self.vel[0], self.vel[1], self.vel[2]
        vx_b = cos_y * vx_w + sin_y * vy_w
        vy_b = -sin_y * vx_w + cos_y * vy_w

        self.accumulated_body_disp[0] += vx_b * dt
        self.accumulated_body_disp[1] += vy_b * dt
        self.accumulated_body_disp[2] += vz_w * dt
        self.accumulated_yaw_rad += self.yaw_rate * dt

        # Evaluate threshold crossing for discrete benchmark macro-actions
        act = ACTION_STOP
        if self.accumulated_yaw_rad >= math.radians(20.0):
            act = ACTION_TURN_LEFT_30
            self.accumulated_yaw_rad -= math.radians(30.0)
            self.yaw = (self.yaw + math.radians(30.0) + math.pi) % (2.0 * math.pi) - math.pi
        elif self.accumulated_yaw_rad <= -math.radians(20.0):
            act = ACTION_TURN_RIGHT_30
            self.accumulated_yaw_rad += math.radians(30.0)
            self.yaw = (self.yaw - math.radians(30.0) + math.pi) % (2.0 * math.pi) - math.pi
        elif self.accumulated_body_disp[2] >= 2.0:
            act = ACTION_CLIMB_3M
            self.accumulated_body_disp[2] -= 3.0
            self.pos[2] += 3.0
        elif self.accumulated_body_disp[2] <= -2.0:
            act = ACTION_DESCEND_3M
            self.accumulated_body_disp[2] += 3.0
            self.pos[2] = max(0.5, self.pos[2] - 3.0)
        elif self.accumulated_body_disp[0] >= 7.5:
            act = ACTION_MOVE_FORWARD_LARGE
            self.accumulated_body_disp[0] -= 9.0
            self.pos[0] += 9.0 * math.cos(self.yaw)
            self.pos[1] += 9.0 * math.sin(self.yaw)
        elif self.accumulated_body_disp[0] >= 4.5:
            act = ACTION_MOVE_FORWARD_MED
            self.accumulated_body_disp[0] -= 6.0
            self.pos[0] += 6.0 * math.cos(self.yaw)
            self.pos[1] += 6.0 * math.sin(self.yaw)
        elif self.accumulated_body_disp[0] >= 2.0:
            act = ACTION_MOVE_FORWARD_FINE
            self.accumulated_body_disp[0] -= 3.0
            self.pos[0] += 3.0 * math.cos(self.yaw)
            self.pos[1] += 3.0 * math.sin(self.yaw)
        elif self.accumulated_body_disp[1] >= 2.0:
            act = ACTION_STRAFE_LEFT_3M
            self.accumulated_body_disp[1] -= 3.0
            self.pos[0] += 3.0 * math.cos(self.yaw + math.pi / 2.0)
            self.pos[1] += 3.0 * math.sin(self.yaw + math.pi / 2.0)
        elif self.accumulated_body_disp[1] <= -2.0:
            act = ACTION_STRAFE_RIGHT_3M
            self.accumulated_body_disp[1] += 3.0
            self.pos[0] += 3.0 * math.cos(self.yaw - math.pi / 2.0)
            self.pos[1] += 3.0 * math.sin(self.yaw - math.pi / 2.0)

        self.last_discrete_action = act
        self.sim_time += dt
