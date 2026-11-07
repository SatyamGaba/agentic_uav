"""
Aerial Vision-Language Navigation Framework.
Modular, swappable adapter architecture for UAV navigation benchmarking.
"""
from typing import Optional
from framework.types import (
    Pose3D,
    VLNObservation,
    NavAction,
    EpisodeGoal,
    StepTelemetry,
    EpisodeResult,
    BenchmarkSummary
)
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_vlm import BaseVLMBackend, VLMResponse
from framework.interfaces.base_env import BaseSimulatorEnv

from framework.perception.transforms import (
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame,
    propagate_body_to_world,
    euclidean_distance_3d
)

from framework.evaluation.metrics import (
    evaluate_episode,
    summarize_benchmark,
    compute_wilson_score_interval
)

from framework.backends.mock_client import MockVLMBackend
from framework.backends.ollama_client import OllamaVLMBackend
from framework.backends.openai_client import OpenAIVLMBackend

from framework.adapters.heuristic import HeuristicRuleAdapter
from framework.adapters.discrete_action import DiscreteActionAdapter
from framework.adapters.direct_pixel import DirectPixelAdapter
from framework.adapters.tool_calling import AgenticNavAdapter
from framework.adapters.waypoint_candidate import WaypointCandidateAdapter
from framework.adapters.calibrated_nav import CalibratedNavAdapter
from framework.adapters.supernav_adapter import SuperNavAdapter

from framework.environments.mock_env import MockSimulatorEnv
from framework.environments.openfly_env import OpenFlyBenchmarkEnv
from framework.environments.airsim_env import AirSimUnrealEnv

from framework.runner import VLNBenchmarkRunner


ADAPTER_REGISTRY = {
    "supernav": SuperNavAdapter,
    "calibrated": CalibratedNavAdapter,
    "calibrated_nav": CalibratedNavAdapter,
    "agentic_nav": AgenticNavAdapter,
    "tool_calling": AgenticNavAdapter,
    "direct_pixel": DirectPixelAdapter,
    "see_point_fly": DirectPixelAdapter,
    "discrete_action": DiscreteActionAdapter,
    "openfly_agent": DiscreteActionAdapter,
    "waypoint_candidate": WaypointCandidateAdapter,
    "smartway": WaypointCandidateAdapter,
    "heuristic": HeuristicRuleAdapter,
}

BACKEND_REGISTRY = {
    "mock": MockVLMBackend,
    "ollama": OllamaVLMBackend,
    "openai": OpenAIVLMBackend,
}

ENV_REGISTRY = {
    "mock": MockSimulatorEnv,
    "openfly": OpenFlyBenchmarkEnv,
    "airsim": AirSimUnrealEnv,
}


def create_adapter(name: str, backend: Optional[BaseVLMBackend] = None, **kwargs) -> BaseVLNAdapter:
    """Factory helper to instantiate any registered adapter."""
    key = name.lower().replace("-", "_")
    if key not in ADAPTER_REGISTRY:
        raise ValueError(f"Unknown adapter '{name}'. Available: {list(ADAPTER_REGISTRY.keys())}")
    cls = ADAPTER_REGISTRY[key]
    if key == "heuristic":
        return cls(**kwargs)
    if backend is None:
        raise ValueError(f"Adapter '{name}' strictly requires a live VLM backend instance.")
    return cls(vlm_backend=backend, **kwargs)


def create_backend(name: str, model_name: str, **kwargs) -> BaseVLMBackend:
    """Factory helper to instantiate any registered VLM backend."""
    key = name.lower()
    if key not in BACKEND_REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Available: {list(BACKEND_REGISTRY.keys())}")
    return BACKEND_REGISTRY[key](model_name=model_name, **kwargs)


def create_environment(name: str, **kwargs) -> BaseSimulatorEnv:
    """Factory helper to instantiate any registered simulation environment."""
    key = name.lower()
    if key not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment '{name}'. Available: {list(ENV_REGISTRY.keys())}")
    return ENV_REGISTRY[key](**kwargs)


__all__ = [
    "Pose3D",
    "VLNObservation",
    "NavAction",
    "EpisodeGoal",
    "StepTelemetry",
    "EpisodeResult",
    "BenchmarkSummary",
    "BaseVLNAdapter",
    "BaseVLMBackend",
    "VLMResponse",
    "BaseSimulatorEnv",
    "CameraIntrinsics",
    "query_metric_depth_patch",
    "backproject_pixel_to_body_frame",
    "propagate_body_to_world",
    "euclidean_distance_3d",
    "evaluate_episode",
    "summarize_benchmark",
    "compute_wilson_score_interval",
    "MockVLMBackend",
    "OllamaVLMBackend",
    "OpenAIVLMBackend",
    "HeuristicRuleAdapter",
    "DiscreteActionAdapter",
    "DirectPixelAdapter",
    "AgenticNavAdapter",
    "WaypointCandidateAdapter",
    "MockSimulatorEnv",
    "OpenFlyBenchmarkEnv",
    "AirSimUnrealEnv",
    "VLNBenchmarkRunner",
    "create_adapter",
    "create_backend",
    "create_environment",
]
