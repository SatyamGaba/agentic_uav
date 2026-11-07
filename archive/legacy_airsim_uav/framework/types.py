"""
Unified Data Contracts for Aerial Vision-Language Navigation Framework.
All adapters, backends, environments, and evaluators communicate via these structures.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import numpy as np


@dataclass
class Pose3D:
    """Represents a 3D pose in world coordinates."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0      # In radians
    pitch: float = 0.0    # In radians
    roll: float = 0.0     # In radians

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]

    def to_dict(self) -> Dict[str, float]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "yaw": round(self.yaw, 4),
            "pitch": round(self.pitch, 4),
            "roll": round(self.roll, 4),
        }


@dataclass
class VLNObservation:
    """Standardized observation payload supplied to VLN adapters."""
    step_id: int
    instruction: str
    pose: Pose3D
    rgb_image: Optional[np.ndarray] = None          # H x W x 3 RGB uint8 array
    rgb_b64: Optional[str] = None                   # Optional base64-encoded JPEG/PNG
    depth_map: Optional[np.ndarray] = None          # H_depth x W_depth metric float32 array
    trajectory_history: List[Pose3D] = field(default_factory=list)
    reasoning_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rgb_shape(self) -> Optional[Tuple[int, int]]:
        return self.rgb_image.shape[:2] if self.rgb_image is not None else None

    @property
    def depth_shape(self) -> Optional[Tuple[int, int]]:
        return self.depth_map.shape[:2] if self.depth_map is not None else None


@dataclass
class NavAction:
    """
    Unified action output from any VLN adapter.
    Supports continuous displacement, discrete action code, or waypoint proposal.
    """
    # 1. Continuous 3D displacement relative to drone body frame:
    # X: Forward (+m), Y: Right (+m), Z: Up (+m), yaw_delta: Radians (+right/-left)
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    yaw_delta: float = 0.0

    # 2. 2D Visual waypoint grounding in normalized screen space [0.0..1.0]:
    u: Optional[float] = None
    v: Optional[float] = None
    depth_m: Optional[float] = None

    # 3. Discrete action code (e.g., OpenFly 10-class action space: 0, 1, 2, 3, 4, 5, 8, 9, -1, -2):
    action_code: Optional[int] = None

    # 4. Status flags and telemetry:
    is_stop: bool = False
    reasoning: str = ""
    confidence: float = 1.0
    latency_s: float = 0.0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    raw_response: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeGoal:
    """Ground truth navigation goal specification."""
    goal_pos: List[float]               # [x, y, z] target
    gt_trajectory: List[List[float]]    # Sequence of ground truth waypoints
    instruction: str
    success_distance: float = 5.0       # Strict threshold for arrival in meters
    coarse_success_distance: float = 20.0 # Official OpenFly / benchmark threshold (20m)
    max_steps: int = 35
    episode_id: Union[int, str] = 0
    scene_id: str = "default_scene"
    env_name: str = "env_airsim_18"
    split: str = "test"
    start_yaw: float = 0.0              # Initial heading in radians
    gt_yaws: List[float] = field(default_factory=list)
    gt_actions: List[int] = field(default_factory=list)


@dataclass
class StepTelemetry:
    """Logged telemetry for each step taken during an episode."""
    step_id: int
    pose: Pose3D
    action: NavAction
    distance_to_goal: float
    step_duration_s: float
    depth_at_waypoint: Optional[float] = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeResult:
    """Comprehensive performance metrics for a single episode."""
    episode_id: Union[int, str]
    instruction: str
    success: bool                       # Strict SR @ 5m
    oracle_success: bool                # Strict OSR @ 5m
    navigation_error: float             # NE (m): distance from final position to goal
    oracle_navigation_error: float      # Min distance achieved along entire trajectory
    gt_length: float                    # GT trajectory length (m)
    pred_length: float                  # Agent predicted trajectory length (m)
    spl: float                          # Strict SPL @ 5m
    steps_taken: int
    duration_s: float
    success_20m: bool = False           # Benchmark SR @ 20m (official OpenFly threshold)
    oracle_success_20m: bool = False    # Benchmark OSR @ 20m
    spl_20m: float = 0.0                # Benchmark SPL @ 20m
    trajectory: List[List[float]] = field(default_factory=list)
    telemetry: List[StepTelemetry] = field(default_factory=list)
    adapter_name: str = "UnknownAdapter"
    model_name: str = "UnknownModel"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "instruction": self.instruction,
            "success_5m": self.success,
            "oracle_success_5m": self.oracle_success,
            "spl_5m": round(self.spl, 4),
            "success_20m": self.success_20m,
            "oracle_success_20m": self.oracle_success_20m,
            "spl_20m": round(self.spl_20m, 4),
            "navigation_error_m": round(self.navigation_error, 2),
            "oracle_ne_m": round(self.oracle_navigation_error, 2),
            "gt_length_m": round(self.gt_length, 2),
            "pred_length_m": round(self.pred_length, 2),
            "steps_taken": self.steps_taken,
            "duration_s": round(self.duration_s, 2),
            "adapter_name": self.adapter_name,
            "model_name": self.model_name,
        }


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark statistics across multiple episodes with 95% Wilson CIs."""
    adapter_name: str
    model_name: str
    environment_name: str
    total_episodes: int
    success_count: int                  # 5m count
    oracle_success_count: int           # 5m count
    success_rate: float                 # SR @ 5m (%)
    sr_ci_95: Tuple[float, float]       # [lower, upper] %
    oracle_success_rate: float          # OSR @ 5m (%)
    osr_ci_95: Tuple[float, float]
    mean_spl: float                     # Mean SPL @ 5m
    success_count_20m: int = 0          # 20m count
    oracle_success_count_20m: int = 0   # 20m count
    success_rate_20m: float = 0.0       # SR @ 20m (%)
    sr_ci_95_20m: Tuple[float, float] = (0.0, 0.0)
    oracle_success_rate_20m: float = 0.0 # OSR @ 20m (%)
    osr_ci_95_20m: Tuple[float, float] = (0.0, 0.0)
    mean_spl_20m: float = 0.0           # Mean SPL @ 20m
    mean_navigation_error: float = 0.0  # Mean NE (m)
    mean_oracle_ne: float = 0.0
    mean_pred_length: float = 0.0
    mean_gt_length: float = 0.0
    mean_duration_s: float = 0.0
    mean_steps: float = 0.0
    timestamp: str = ""
    episodes: List[EpisodeResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "adapter_name": self.adapter_name,
            "model_name": self.model_name,
            "environment_name": self.environment_name,
            "total_episodes": self.total_episodes,
            "sr_5m_pct": round(self.success_rate, 2),
            "sr_ci_95_5m": [round(self.sr_ci_95[0], 2), round(self.sr_ci_95[1], 2)],
            "osr_5m_pct": round(self.oracle_success_rate, 2),
            "osr_ci_95_5m": [round(self.osr_ci_95[0], 2), round(self.osr_ci_95[1], 2)],
            "mean_spl_5m": round(self.mean_spl, 4),
            "sr_20m_pct": round(self.success_rate_20m, 2),
            "sr_ci_95_20m": [round(self.sr_ci_95_20m[0], 2), round(self.sr_ci_95_20m[1], 2)],
            "osr_20m_pct": round(self.oracle_success_rate_20m, 2),
            "osr_ci_95_20m": [round(self.osr_ci_95_20m[0], 2), round(self.osr_ci_95_20m[1], 2)],
            "mean_spl_20m": round(self.mean_spl_20m, 4),
            "mean_navigation_error_m": round(self.mean_navigation_error, 2),
            "mean_oracle_ne_m": round(self.mean_oracle_ne, 2),
            "mean_pred_length_m": round(self.mean_pred_length, 2),
            "mean_gt_length_m": round(self.mean_gt_length, 2),
            "mean_duration_s": round(self.mean_duration_s, 2),
            "mean_steps": round(self.mean_steps, 1),
            "timestamp": self.timestamp
        }
