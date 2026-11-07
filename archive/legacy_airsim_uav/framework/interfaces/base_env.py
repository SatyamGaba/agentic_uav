"""
Abstract Base Class for Simulation Environments.
Allows identical benchmark execution across AirSim/UE4, OpenFly discrete kinematics, and Mock 3D.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
from framework.types import VLNObservation, NavAction, EpisodeGoal


class BaseSimulatorEnv(ABC):
    """
    Abstract simulator environment interface for Aerial VLN.
    """
    def __init__(self, name: str = "BaseSimulatorEnv", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def reset(self, goal: EpisodeGoal) -> VLNObservation:
        """
        Resets simulator to the episode's initial state and returns initial observation.
        """
        pass

    @abstractmethod
    def step(self, action: NavAction) -> Tuple[VLNObservation, bool, Dict[str, Any]]:
        """
        Executes the navigation action in the simulator.
        Returns:
            - next_observation: VLNObservation
            - done: bool (True if terminal state reached, goal achieved, or collision)
            - info: Dict[str, Any] (telemetry, distance to goal, collision flags)
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleans up simulator resources and connections."""
        pass
