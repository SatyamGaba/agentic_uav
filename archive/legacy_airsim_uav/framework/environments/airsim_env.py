"""
AirSim / Unreal Engine 4 Closed-Loop Simulation Environment.
Handles live multirotor connection, sensor capture with resolution decoupling,
PIL aspect-safe image encoding, and 3D flight execution.
"""
import io
import time
import base64
import math
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any, Optional, List
from framework.interfaces.base_env import BaseSimulatorEnv
from framework.types import VLNObservation, NavAction, EpisodeGoal, Pose3D
from framework.perception.transforms import euclidean_distance_3d

try:
    import airsim
    HAS_AIRSIM = True
except ImportError:
    HAS_AIRSIM = False


class AirSimUnrealEnv(BaseSimulatorEnv):
    """
    Closed-loop AirSim / UE4 simulation environment.
    """
    def __init__(
        self,
        ip: str = "127.0.0.1",
        port: int = 41451,
        vehicle_name: str = "drone_1",
        camera_name: str = "front_custom",
        vlm_img_size: Tuple[int, int] = (384, 216),
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="AirSimUnrealEnv", config=config)
        self.ip = ip
        self.port = port
        self.vehicle_name = vehicle_name
        self.camera_name = camera_name
        self.vlm_img_size = vlm_img_size
        self.client = None

        self.current_goal: Optional[EpisodeGoal] = None
        self.step_count = 0
        self.trajectory_history: List[Pose3D] = []

    def connect(self) -> bool:
        if not HAS_AIRSIM:
            return False
        try:
            self.client = airsim.MultirotorClient(ip=self.ip, port=self.port)
            self.client.confirmConnection()
            self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
            self.client.armDisarm(True, vehicle_name=self.vehicle_name)
            return True
        except Exception:
            self.client = None
            return False

    def reset(self, goal: EpisodeGoal) -> VLNObservation:
        self.current_goal = goal
        self.step_count = 0
        self.trajectory_history = []

        if self.client is None:
            self.connect()

        if self.client:
            try:
                self.client.reset()
                self.client.enableApiControl(True, vehicle_name=self.vehicle_name)
                self.client.armDisarm(True, vehicle_name=self.vehicle_name)
                # Teleport to start position safely above ground/rooftop with true start yaw
                start = goal.gt_trajectory[0] if goal.gt_trajectory else [0.0, 0.0, 5.0]
                safe_z = -max(2.5, abs(start[2]))
                init_pos = airsim.Vector3r(start[0], -start[1], safe_z)
                self.client.simSetVehiclePose(
                    airsim.Pose(init_pos, airsim.to_quaternion(0, 0, -goal.start_yaw)),
                    True,
                    vehicle_name=self.vehicle_name
                )
                time.sleep(0.5)
            except Exception:
                self.connect()

        return self._capture_observation()

    def step(self, action: NavAction) -> Tuple[VLNObservation, bool, Dict[str, Any]]:
        self.step_count += 1
        shield_activated = False

        if action.is_stop:
            obs = self._capture_observation()
            dist = euclidean_distance_3d(obs.pose.to_list(), self.current_goal.goal_pos)
            done = True
            info = {
                "reason": "Agent stopped",
                "distance_to_goal": dist,
                "step_count": self.step_count,
                "is_success_5m": dist <= self.current_goal.success_distance,
                "is_success_20m": dist <= self.current_goal.coarse_success_distance,
                "safety_shield_activated": False
            }
            return obs, done, info

        if self.client:
            try:
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                pos = state.kinematics_estimated.position
                curr_yaw = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]

                # System 1 Reactive Safety Shield: inspect obstacle clearance to prevent PhysX SIGSEGV
                safe_dx = action.dx
                safe_dy = action.dy
                safe_dz = action.dz

                if action.depth_m is not None and action.depth_m < 3.2:
                    # Near obstacle: limit forward penetration, apply climb clearance
                    shield_activated = True
                    safe_dx = min(safe_dx, max(0.5, action.depth_m - 1.2))
                    safe_dz = max(safe_dz, 1.5)

                mid_yaw = curr_yaw + action.yaw_delta * 0.5
                target_x = pos.x_val + (safe_dx * math.cos(mid_yaw) - safe_dy * math.sin(mid_yaw))
                target_y = pos.y_val + (safe_dx * math.sin(mid_yaw) + safe_dy * math.cos(mid_yaw))
                target_z = pos.z_val - safe_dz

                # Maintain safety altitude floor above rooftop
                if target_z > -1.0:
                    target_z = -2.0

                self.client.moveToPositionAsync(
                    target_x,
                    target_y,
                    target_z,
                    velocity=2.5,
                    yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=math.degrees(curr_yaw + action.yaw_delta)),
                    vehicle_name=self.vehicle_name
                ).join()
            except Exception:
                self.connect()

        obs = self._capture_observation()
        dist = euclidean_distance_3d(obs.pose.to_list(), self.current_goal.goal_pos)
        is_success_5m = dist <= self.current_goal.success_distance
        is_success_20m = dist <= self.current_goal.coarse_success_distance
        is_timeout = self.step_count >= self.current_goal.max_steps

        done = is_success_5m or is_timeout
        info = {
            "distance_to_goal": dist,
            "is_success_5m": is_success_5m,
            "is_success_20m": is_success_20m,
            "step_count": self.step_count,
            "timeout": is_timeout,
            "safety_shield_activated": shield_activated
        }
        return obs, done, info

    def _capture_observation(self) -> VLNObservation:
        rgb_arr = None
        b64_img = None
        depth_arr = None
        pose = Pose3D()

        if self.client:
            try:
                state = self.client.getMultirotorState(vehicle_name=self.vehicle_name)
                p = state.kinematics_estimated.position
                y = airsim.to_eularian_angles(state.kinematics_estimated.orientation)[2]
                pose = Pose3D(x=p.x_val, y=p.y_val, z=-p.z_val, yaw=y)

                # Request RGB and Depth in a single batch
                requests = [
                    airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False),
                    airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)
                ]
                responses = self.client.simGetImages(requests, vehicle_name=self.vehicle_name)

                # 1. Process RGB (1920x1080)
                raw_rgb = responses[0].image_data_uint8
                rw, rh = responses[0].width, responses[0].height
                if rw > 0 and rh > 0 and len(raw_rgb) > 0:
                    full_img = Image.frombytes("RGB", (rw, rh), raw_rgb)
                    rgb_arr = np.array(full_img)

                    # Aspect-safe downsample for VLM
                    resized = full_img.resize(self.vlm_img_size, Image.Resampling.BILINEAR)
                    buf = io.BytesIO()
                    resized.save(buf, format="JPEG", quality=85)
                    b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

                # 2. Process Depth (640x480 float planar)
                dw, dh = responses[1].width, responses[1].height
                if dw > 0 and dh > 0 and len(responses[1].image_data_float) > 0:
                    depth_arr = np.array(responses[1].image_data_float, dtype=np.float32).reshape(dh, dw)

            except Exception:
                pass

        if not self.trajectory_history or pose != self.trajectory_history[-1]:
            self.trajectory_history.append(pose)

        return VLNObservation(
            step_id=self.step_count,
            instruction=self.current_goal.instruction if self.current_goal else "",
            pose=pose,
            rgb_image=rgb_arr,
            rgb_b64=b64_img,
            depth_map=depth_arr,
            trajectory_history=list(self.trajectory_history),
            metadata={"airsim_live": True}
        )

    def close(self) -> None:
        if self.client:
            try:
                self.client.enableApiControl(False, vehicle_name=self.vehicle_name)
            except Exception:
                pass
            self.client = None
