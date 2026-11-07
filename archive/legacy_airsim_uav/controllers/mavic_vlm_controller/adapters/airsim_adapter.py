"""
AirSim Continuous Visual Servoing Adapter
=========================================
Bridges continuous 6-DoF velocity & gimbal attitude to AirSim RPC clients.
Maintains strict coordinate frame parity between robotics World ENU (+X Forward, +Y Left, +Z Up)
and Microsoft AirSim NED (+X North, +Y East, +Z Down).

Supports both live RPC flight and offline mock testing with verified contracts.
"""

from __future__ import annotations
import math
import logging
from typing import Tuple, Optional, Any
import numpy as np

from contracts import DroneTelemetry, CameraIntrinsics
from adapters.base_adapter import ContinuousUAVController

logger = logging.getLogger("MavicVLM")

try:
    import airsim
    HAS_AIRSIM = True
except ImportError:
    HAS_AIRSIM = False


class AirSimAdapter(ContinuousUAVController):
    """Adapter for continuous 6-DoF flight in Microsoft AirSim."""

    def __init__(
        self,
        airsim_client: Optional[Any] = None,
        initial_pos: Optional[np.ndarray] = None,
        initial_yaw: float = 0.0,
        vehicle_name: str = "drone_1",
        camera_name: str = "front_custom",
    ):
        self.client = airsim_client
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name

        # Internal pose maintained in World ENU (+X East/Forward, +Y North/Left, +Z Up)
        self.pos = initial_pos.copy() if initial_pos is not None else np.array([0.0, 0.0, 5.0], dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = initial_yaw
        self.yaw_rate = 0.0
        self.gimbal_pitch = 0.0
        self.gimbal_yaw = 0.0
        self.sim_time = 0.0

        # Calibrated 640x360 camera intrinsics matching AirSim settings.json (HFOV=90.0 deg)
        # HFOV = 90.0 deg -> focal length fx = (640 / 2) / tan(90 deg / 2) = 320.0px
        fx = (640.0 / 2.0) / math.tan(math.radians(90.0) / 2.0)
        fy = fx
        cx = 320.0
        cy = 180.0
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        vfov_rad = 2.0 * math.atan((360.0 / 2.0) / fy)
        self.intrinsics = CameraIntrinsics(
            K=K, width=640, height=360, vfov=vfov_rad, hfov=math.radians(90.0)
        )

        # Injected perception cache for offline simulation / testing
        self._cached_rgb: Optional[np.ndarray] = None
        self._cached_depth: Optional[np.ndarray] = None
        self.camera_mount_pos = None


    # -----------------------------------------------------------------------
    # Coordinate Transforms: World ENU <-> AirSim NED
    # -----------------------------------------------------------------------

    @staticmethod
    def enu_to_airsim_ned(pos_enu: np.ndarray) -> np.ndarray:
        """Transforms position/velocity from World ENU to AirSim NED.
        
        ENU:  X_enu (Forward/East), Y_enu (Left/North), Z_enu (Up)
        NED:  X_ned = X_enu, Y_ned = -Y_enu, Z_ned = -Z_enu
        """
        return np.array([pos_enu[0], -pos_enu[1], -pos_enu[2]], dtype=np.float64)

    @staticmethod
    def airsim_ned_to_enu(pos_ned: np.ndarray) -> np.ndarray:
        """Transforms position/velocity from AirSim NED to World ENU.
        
        NED:  X_ned (North), Y_ned (East), Z_ned (Down)
        ENU:  X_enu = X_ned, Y_enu = -Y_ned, Z_enu = -Z_ned
        """
        return np.array([pos_ned[0], -pos_ned[1], -pos_ned[2]], dtype=np.float64)

    @staticmethod
    def enu_yaw_to_airsim(yaw_enu: float) -> float:
        """Transforms yaw angle from ENU to AirSim NED (reverses rotation direction)."""
        return -yaw_enu

    @staticmethod
    def airsim_yaw_to_enu(yaw_ned: float) -> float:
        """Transforms yaw angle from AirSim NED to ENU."""
        return -yaw_ned

    # -----------------------------------------------------------------------
    # Continuous Control Interface
    # -----------------------------------------------------------------------

    def set_continuous_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        """Sets target continuous velocity [m/s] in World ENU and yaw rate [rad/s]."""
        self.vel = np.array([vx, vy, vz], dtype=np.float64)
        self.yaw_rate = yaw_rate

    def set_gimbal_angles(self, pitch: float, yaw: float = 0.0) -> None:
        """Commands 2-DoF camera gimbal attitude while preserving mount translation offset."""
        self.gimbal_pitch = pitch
        self.gimbal_yaw = yaw
        if self.client and HAS_AIRSIM:
            try:
                if self.camera_mount_pos is None:
                    try:
                        info = self.client.simGetCameraInfo(self.camera_name, vehicle_name=self.vehicle_name)
                        self.camera_mount_pos = info.pose.position
                    except Exception:
                        self.camera_mount_pos = airsim.Vector3r(0.25, 0.0, -0.12)

                # AirSim gimbal orientation is in NED angles
                pitch_ned = -pitch
                yaw_ned = -yaw
                cam_pose = airsim.Pose(
                    self.camera_mount_pos,
                    airsim.to_quaternion(pitch_ned, 0.0, yaw_ned)
                )
                self.client.simSetCameraPose(
                    self.camera_name,
                    cam_pose,
                    vehicle_name=self.vehicle_name
                )
            except Exception as e:
                logger.warning(f"[AirSimAdapter] Gimbal actuation failed: {e}")


    def get_telemetry(self) -> DroneTelemetry:
        """Retrieves vehicle telemetry in World ENU coordinates."""
        if self.client and HAS_AIRSIM:
            try:
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                p = state.kinematics_estimated.position
                v = state.kinematics_estimated.linear_velocity
                q = state.kinematics_estimated.orientation
                yaw_ned = airsim.to_eularian_angles(q)[2]

                pos_enu = self.airsim_ned_to_enu(np.array([p.x_val, p.y_val, p.z_val]))
                vel_enu = self.airsim_ned_to_enu(np.array([v.x_val, v.y_val, v.z_val]))
                yaw_enu = self.airsim_yaw_to_enu(yaw_ned)

                return DroneTelemetry(
                    position=pos_enu,
                    velocity=vel_enu,
                    yaw=yaw_enu,
                    gimbal_pitch=self.gimbal_pitch,
                    gimbal_yaw=self.gimbal_yaw,
                    timestamp=state.timestamp / 1e9,
                )
            except Exception as e:
                logger.warning(f"[AirSimAdapter] get_telemetry RPC failed: {e}")

        # Fallback to internal kinematics state
        return DroneTelemetry(
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            yaw=self.yaw,
            gimbal_pitch=self.gimbal_pitch,
            gimbal_yaw=self.gimbal_yaw,
            timestamp=self.sim_time,
        )

    def get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        """Retrieves RGB and metric planar depth observations."""
        if self.client and HAS_AIRSIM:
            try:
                requests = [
                    airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPlanar, True, False)
                ]
                responses = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)
                if responses and len(responses) >= 2:
                    # 1. RGB
                    raw_rgb = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                    rgb = raw_rgb.reshape(responses[0].height, responses[0].width, 3)
                    
                    # 2. Depth Planar (float32 meters)
                    depth = np.array(responses[1].image_data_float, dtype=np.float32).reshape(
                        responses[1].height, responses[1].width
                    )
                    return rgb, depth, self.intrinsics
            except Exception as e:
                logger.warning(f"[AirSimAdapter] simGetImages RPC failed: {e}")

        # Return cached observation if injected for testing
        rgb = self._cached_rgb if self._cached_rgb is not None else np.zeros((360, 640, 3), dtype=np.uint8)
        depth = self._cached_depth if self._cached_depth is not None else np.full((360, 640), 20.0, dtype=np.float32)
        return rgb, depth, self.intrinsics

    def set_cached_observation(self, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Injects test observations for offline replay or unit testing."""
        self._cached_rgb = rgb
        self._cached_depth = depth

    def step_simulation(self, dt: float) -> None:
        """Advances physical simulation by dt seconds."""
        if self.client and HAS_AIRSIM:
            try:
                # Convert continuous velocity vector to AirSim NED
                v_ned = self.enu_to_airsim_ned(self.vel)
                yaw_rate_deg = math.degrees(self.enu_yaw_to_airsim(self.yaw_rate))

                # Issue non-blocking velocity command
                self.client.moveByVelocityAsync(
                    float(v_ned[0]),
                    float(v_ned[1]),
                    float(v_ned[2]),
                    duration=float(dt),
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate_deg),
                    vehicle_name=self.vehicle_name
                )
            except Exception as e:
                logger.warning(f"[AirSimAdapter] moveByVelocityAsync failed: {e}")

        # Integrate internal state for dead-reckoning / mock mode
        self.pos += self.vel * dt
        self.yaw += self.yaw_rate * dt
        # Normalize yaw in [-pi, pi]
        self.yaw = (self.yaw + math.pi) % (2.0 * math.pi) - math.pi
        self.sim_time += dt
