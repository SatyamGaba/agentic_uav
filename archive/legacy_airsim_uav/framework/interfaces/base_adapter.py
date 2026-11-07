"""
Abstract Base Class for all Vision-Language Navigation Adapters.
Any framework implementation (AgenticNav, See-Point-Fly, OpenFly-Agent, SmartWay, etc.)
inherits from BaseVLNAdapter.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from framework.types import VLNObservation, NavAction


class BaseVLNAdapter(ABC):
    """
    Abstract interface for swappable Aerial VLN decision adapters.
    """
    def __init__(self, name: str = "BaseVLNAdapter", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}

    @abstractmethod
    def reset(self, initial_observation: VLNObservation) -> None:
        """
        Resets adapter state at the beginning of each episode.
        Clears memory buffers, history, or internal states.
        """
        pass

    @abstractmethod
    def step(self, observation: VLNObservation) -> NavAction:
        """
        Computes the next navigation action given the current multi-modal observation.
        Must return a standardized NavAction.
        """
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        """Returns internal state telemetry and diagnostics for debugging."""
        return {
            "adapter_name": self.name,
            "config": self.config
        }
