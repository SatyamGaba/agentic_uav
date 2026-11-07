"""
Synthetic 3D Continuous Simulation Environment for Local Verification & CI.
Generates synthetic RGB frames and metric depth maps with realistic target geometry,
allowing end-to-end validation of all adapters without external simulator dependencies.
"""
import math
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from framework.interfaces.base_env import BaseSimulatorEnv
from framework.types import VLNObservation, NavAction, EpisodeGoal, Pose3D
from framework.perception.transforms import propagate_body_to_world, euclidean_distance_3d


class MockSimulatorEnv(BaseSimulatorEnv):
    """
    Fast, deterministic synthetic 3D continuous environment.
    Simulates drone translation and rotation in 3D Euclidean space.
    Renders realistic synthetic depth maps with target landmarks.
    """
    def __init__(
        self,
        rgb_width: int = 640,
        rgb_height: int = 360,
        depth_width: int = 640,
        depth_height: int = 480,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="MockSimulatorEnv", config=config)
        self.rgb_w = rgb_width
        self.rgb_h = rgb_height
        self.depth_w = depth_width
        self.depth_h = depth_height

        self.current_pose = Pose3D()
        self.current_goal: Optional[EpisodeGoal] = None
        self.step_count = 0
        self.trajectory_history: List[Pose3D] = []

    def reset(self, goal: EpisodeGoal) -> VLNObservation:
        self.current_goal = goal
        self.step_count = 0

        # Start at origin or goal's start position
        start_x = goal.gt_trajectory[0][0] if goal.gt_trajectory else 0.0
        start_y = goal.gt_trajectory[0][1] if goal.gt_trajectory else 0.0
        start_z = goal.gt_trajectory[0][2] if goal.gt_trajectory else 5.0

        # Face toward goal initially
        dx = goal.goal_pos[0] - start_x
        dy = goal.goal_pos[1] - start_y
        initial_yaw = math.atan2(dy, dx)

        self.current_pose = Pose3D(x=start_x, y=start_y, z=start_z, yaw=initial_yaw)
        self.trajectory_history = [Pose3D(x=start_x, y=start_y, z=start_z, yaw=initial_yaw)]

        return self._build_observation()

    def step(self, action: NavAction) -> Tuple[VLNObservation, bool, Dict[str, Any]]:
        self.step_count += 1

        # Check explicit stop action
        if action.is_stop:
            dist_to_goal = euclidean_distance_3d(self.current_pose.to_list(), self.current_goal.goal_pos)
            done = True
            info = {
                "reason": "Agent declared arrival (STOP)",
                "distance_to_goal": dist_to_goal,
                "step_count": self.step_count
            }
            return self._build_observation(), done, info

        # Propagate kinematics in world coordinates
        next_x, next_y, next_z, next_yaw = propagate_body_to_world(
            current_x=self.current_pose.x,
            current_y=self.current_pose.y,
            current_z=self.current_pose.z,
            current_yaw_rad=self.current_pose.yaw,
            dx_body=action.dx,
            dy_body=action.dy,
            dz_body=action.dz,
            yaw_delta_rad=action.yaw_delta
        )

        self.current_pose = Pose3D(x=next_x, y=next_y, z=next_z, yaw=next_yaw)
        self.trajectory_history.append(Pose3D(x=next_x, y=next_y, z=next_z, yaw=next_yaw))

        dist_to_goal = euclidean_distance_3d(self.current_pose.to_list(), self.current_goal.goal_pos)
        is_success = dist_to_goal <= self.current_goal.success_distance
        is_timeout = self.step_count >= self.current_goal.max_steps

        done = is_success or is_timeout
        info = {
            "distance_to_goal": dist_to_goal,
            "is_success": is_success,
            "step_count": self.step_count,
            "timeout": is_timeout
        }

        return self._build_observation(), done, info

    def _build_observation(self) -> VLNObservation:
        # Generate synthetic depth map (distance gradient + target object landmark)
        dist_to_goal = euclidean_distance_3d(self.current_pose.to_list(), self.current_goal.goal_pos) if self.current_goal else 25.0

        depth_map = np.full((self.depth_h, self.depth_w), 15.0, dtype=np.float32)
        # Add target landmark at center if approaching
        if dist_to_goal < 15.0:
            cx, cy = self.depth_w // 2, self.depth_h // 2
            r = max(5, int(50.0 / max(1.0, dist_to_goal)))
            depth_map[max(0, cy - r): min(self.depth_h, cy + r), max(0, cx - r): min(self.depth_w, cx + r)] = float(dist_to_goal)

        # Generate synthetic RGB frame (H x W x 3)
        rgb_image = np.zeros((self.rgb_h, self.rgb_w, 3), dtype=np.uint8)
        rgb_image[:, :] = [135, 206, 235]  # Sky blue
        rgb_image[self.rgb_h // 2:, :] = [34, 139, 34]  # Ground green

        return VLNObservation(
            step_id=self.step_count,
            instruction=self.current_goal.instruction if self.current_goal else "",
            pose=Pose3D(
                x=self.current_pose.x,
                y=self.current_pose.y,
                z=self.current_pose.z,
                yaw=self.current_pose.yaw
            ),
            rgb_image=rgb_image,
            depth_map=depth_map,
            trajectory_history=list(self.trajectory_history),
            metadata={"mock_sim": True, "dist_to_goal": dist_to_goal}
        )

    def close(self) -> None:
        pass
