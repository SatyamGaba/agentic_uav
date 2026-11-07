"""Abstract embodiment base class with strict privilege firewall enforcement."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class StepObservation:
    """Observations strictly allowed across the privilege firewall to the agent."""
    rgb_views: Dict[int, np.ndarray]      # 4 views: 0=Front, 1=Right, 2=Back, 3=Left
    depth_views: Dict[int, np.ndarray]    # 4 depth buffers (accessed only via tools)
    allocentric_pose: Tuple[float, float, float]  # (x, z, yaw_rad) relative to start pose
    simulator_actions_taken: int
    is_done: bool
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    primitive_trajectory: List[Tuple[float, float, float]] = field(default_factory=list)
    primitive_geodesic_distances: List[float] = field(default_factory=list)
    actual_distance_moved: float = 0.0
    collided: bool = False



class EmbodimentBase(ABC):
    """Abstract embodiment interface separating agent harness from simulator internals.

    Privilege Firewall Guarantees:
        - The agent harness NEVER receives ground-truth goal coordinates or scene NavMesh.
        - The agent receives only raw panoramic RGB-D observations and odometry pose.
        - Geodesic evaluation metrics are evaluated by the external harness at episode end.
    """

    @abstractmethod
    def reset(self, episode_data: Dict[str, Any]) -> StepObservation:
        """Reset environment to the episode start pose and return initial observation."""
        pass

    @abstractmethod
    def step_discrete(self, turn_steps_15deg: int, forward_steps_025m: int) -> StepObservation:
        """Execute quantized low-level turn and forward steps.

        Args:
            turn_steps_15deg: Integer turn count (+left CCW, -right CW).
            forward_steps_025m: Integer forward step count in 0.25m increments.

        Returns:
            StepObservation after executing all actions.
        """
        pass

    @abstractmethod
    def get_geodesic_distance_to_goal(self) -> float:
        """Privileged oracle query for evaluation harness only. Never exposed to agent prompt."""
        pass

    @abstractmethod
    def get_shortest_path_distance(self) -> float:
        """Privileged oracle query for evaluation harness only. Never exposed to agent prompt."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Clean up simulator instance."""
        pass

