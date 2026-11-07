"""
Iteration 6 Data Contracts & Interface Schemas
=============================================
Single source of truth for all data interchange between System 1 (Fast Loop),
System 2 (Slow Deliberative Loop), and Pluggable Adapters.

Coordinate Conventions:
- World Frame: ENU (East-North-Up), metric.
    +X: East (meters)
    +Y: North (meters)
    +Z: Up (Altitude above ground, meters)
- Body Frame: FLU (Forward-Left-Up), metric.
    +X: Forward (nose of the UAV)
    +Y: Left (port side)
    +Z: Up (dorsal)
- Camera Optical Frame: RDF (Right-Down-Forward).
    +X: Right across image plane
    +Y: Down across image plane
    +Z: Forward along optical optical axis
- Yaw: Radians, measured counter-clockwise from East (+X) toward North (+Y).
- Gimbal Pitch: Radians, 0.0 = horizontal, positive = tilted down towards nadir.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple, List, Literal, Any, Dict
import numpy as np
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Telemetry & Intrinsics
# ---------------------------------------------------------------------------

@dataclass
class DroneTelemetry:
    """State vector of the UAV at a specific timestamp.
    
    All spatial coordinates are in metric World ENU.
    """
    position: np.ndarray             # [x, y, z] in World ENU (meters)
    velocity: np.ndarray             # [vx, vy, vz] in World ENU (m/s)
    yaw: float                       # Heading in radians [-π, π], CCW from East
    pitch: float = 0.0               # Airframe pitch in radians
    roll: float = 0.0                # Airframe roll in radians
    yaw_rate: float = 0.0            # Heading rate (rad/s)
    gimbal_pitch: float = 0.0        # Gimbal pitch in radians [0.0 = horizon, >0 down]
    gimbal_yaw: float = 0.0          # Gimbal azimuth relative to body in radians
    timestamp: float = 0.0           # Monotonic seconds

    @property
    def speed(self) -> float:
        """Groundspeed magnitude in m/s."""
        return float(np.linalg.norm(self.velocity[:2]))

    @property
    def altitude(self) -> float:
        """Altitude +Z above ground datum in meters."""
        return float(self.position[2])


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic calibration parameters."""
    K: np.ndarray                    # 3x3 camera matrix
    width: int                       # Image width in pixels
    height: int                      # Image height in pixels
    vfov: float                      # Vertical Field of View (radians)
    hfov: float                      # Horizontal Field of View (radians)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def fy(self) -> float:
        return float(self.K[1, 1])

    @property
    def cx(self) -> float:
        return float(self.K[0, 2])

    @property
    def cy(self) -> float:
        return float(self.K[1, 2])

    def unproject_pixel(self, u: float, v: float, depth_m: float) -> np.ndarray:
        """Unprojects image pixel (u, v) with depth into 3D Camera RDF frame."""
        x_c = (u - self.cx) * depth_m / self.fx
        y_c = (v - self.cy) * depth_m / self.fy
        z_c = depth_m
        return np.array([x_c, y_c, z_c], dtype=np.float64)


# ---------------------------------------------------------------------------
# Cognitive Subgoals (System 2 -> System 1)
# ---------------------------------------------------------------------------

class SubgoalIntent(str, Enum):
    """Dynamic high-level navigation intents arbitrated by System 2."""
    SEARCH_AND_ORIENT = "INTENT_SEARCH_AND_ORIENT"
    CORRIDOR_TRANSIT  = "INTENT_CORRIDOR_TRANSIT"
    TARGET_APPROACH   = "INTENT_TARGET_APPROACH"
    SURVEY_INSPECT    = "INTENT_SURVEY_INSPECT"
    TERMINAL_VERIFY   = "INTENT_TERMINAL_VERIFY"


class Subgoal(BaseModel):
    """Structured directive emitted by System 2 for System 1 execution."""
    subgoal_id: int = 1
    intent: SubgoalIntent
    waypoint_xyz: Tuple[float, float, float]   # Target 3D coordinate in World ENU
    target_node_id: Optional[int] = None       # Active 3D Scene Graph node id if tracking
    v_target: float = 3.0                      # Target cruise speed in m/s (clamped by APF)
    caution_level: Literal["low", "medium", "high"] = "medium"
    termination_condition: str = "arrival_within_2m"
    issued_at: float = 0.0                     # Monotonic timestamp of emission
    expiry_s: float = 6.0                      # Heartbeat timeout triggering failsafe
    detector_queries: List[str] = Field(default_factory=list)  # Open-vocab search queries for SoM


# ---------------------------------------------------------------------------
# 3D Metric Scene Graph
# ---------------------------------------------------------------------------

@dataclass
class SceneNode:
    """A grounded 3D landmark in the metric scene graph.
    
    Zero hardcoded volumetric priors: centroid and dimensions are measured
    directly from physical depth point clouds.
    """
    node_id: int
    label: str
    centroid_world: np.ndarray       # [X, Y, Z] in World ENU meters
    dimensions_3d: np.ndarray        # [sx, sy, sz] physical span in meters
    confidence: float                # Fused confidence score in [0.0, 1.0]
    observation_count: int           # Number of multi-view observations
    last_observed_timestamp: float   # Monotonic time of most recent observation
    first_viewpoint: np.ndarray      # UAV position from first observation
    is_confirmed: bool               # True if multi-view baseline passed (anti-hallucination)
    yaw_box: float = 0.0             # Principal planar orientation in radians
    parent_structure_id: Optional[int] = None  # Hierarchical hook (e.g. building for a window)


# ---------------------------------------------------------------------------
# Set-of-Mark (SoM) Grounding
# ---------------------------------------------------------------------------

@dataclass
class MarkedCandidate:
    """A visual candidate proposal tagged with a Set-of-Mark ID."""
    mark_id: int
    bbox_norm: Tuple[float, float, float, float]  # [u_min, v_min, u_max, v_max] normalized in [0, 1]
    label_guess: str
    confidence: float = 0.0
    scene_node_id: Optional[int] = None           # Associated SceneNode if already matched


class MarkSelection(BaseModel):
    """VLM's discrete semantic selection among numbered Set-of-Mark candidates."""
    selected_mark_id: Optional[int] = None        # None = "None of the above / keep searching"
    confidence: float = 0.0                       # Subjective confidence [0.0, 1.0]
    reasoning: str = ""                           # Chain-of-thought grounding rationale

    @model_validator(mode="before")
    @classmethod
    def validate_schema_payload(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # If the model echoed the JSON schema format by nesting under 'properties', unwrap it
            if "properties" in data and isinstance(data["properties"], dict):
                data = data["properties"]
            # Strict schema validation: if dictionary contains completely unrecognized keys (e.g. raw bbox dict), reject
            if not any(k in data for k in ("selected_mark_id", "confidence", "reasoning")):
                raise ValueError(f"Unrecognized schema payload for MarkSelection: keys={list(data.keys())}")
        return data


# ---------------------------------------------------------------------------
# Target Memory Tracking & Recovery
# ---------------------------------------------------------------------------

class TargetRecoveryState(str, Enum):
    TRACKING                = "TRACKING"
    EDGE_ANTICIPATION       = "EDGE_ANTICIPATION"
    METRIC_3D_REACQUISITION = "METRIC_3D_REACQUISITION"
    BUDGETED_SEARCH         = "BUDGETED_SEARCH"
    LOST_DECLARED           = "LOST_DECLARED"


@dataclass
class TargetRecoveryAction:
    """Low-level reaction commanded when a visual tracking target approaches frame boundaries or is lost."""
    gimbal_pitch_cmd: float
    gimbal_yaw_cmd: float
    yaw_rate_cmd: float
    velocity_cmd: Optional[np.ndarray] = None      # 3D velocity override in World ENU
    trigger_replan: bool = False                   # Signal System 2 for asynchronous re-planning


# ---------------------------------------------------------------------------
# Safety Fallback State Machine
# ---------------------------------------------------------------------------

class SafetyState(str, Enum):
    """Deterministic safety states for low-level flight preservation."""
    NORMAL              = "NORMAL"
    EMERGENCY_REPULSION = "EMERGENCY_REPULSION"
    REACQUISITION       = "REACQUISITION"
    BUDGETED_SEARCH     = "BUDGETED_SEARCH"
    HOVER_REPLAN        = "HOVER_REPLAN"
