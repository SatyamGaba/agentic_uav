"""Configuration parameters for AgenticNav replication (v7 Specification)."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ReplicaConfig:
    # Sensor parameters [?]
    image_width: int = 640
    image_height: int = 480
    hfov_deg: float = 90.0
    d_min: float = 0.5   # meters
    d_max: float = 5.0   # meters
    camera_height: float = 1.25  # meters above floor contact datum (y=0)

    # Robot geometry & Kinematics [P / I / ?]
    robot_radius: float = 0.18        # r_a (m) [P]
    base_margin: float = 0.30         # m (m) [?]
    rho_max: float = 2.5              # max forward displacement (m) [?]
    turn_step_deg: float = 15.0       # turn_increment_deg (deg) [P]
    forward_step_m: float = 0.25      # discrete forward step (m) [P]

    # Standoff and snapping ablation switches (Section 11, default off)
    standoff_mode: str = "fixed"      # "fixed" (default), "slope" (ablation only)
    adjacent_bin_snap: bool = False   # False (default, not in paper), True (ablation only)
    max_snap_deg: float = 9.0
    max_snap_drift_m: float = 0.35

    # Zero rho policy (Section 5.3)
    zero_rho_policy: str = "rotate"   # "rotate" (default, turn-only move), "reject" (no_motion)

    # Obstacle source (Section 5.3.1)
    obstacle_source: str = "current_frame"  # "current_frame" (default), "accumulated_grid", "current_plus_near_cache"

    # Obstacle height filter [?]
    chassis_y_min: float = 0.10       # y_min (m)
    chassis_y_max: float = 1.25       # y_max (m)

    # BEV Mapping [P / ?]
    bev_grid_size: int = 256
    bev_resolution_m: float = 0.15

    # Memory & Multi-turn budgets [P / I / ?]
    max_history_steps: int = 6
    max_evidence_queries_per_step: int = 4  # E_max [P]
    max_consecutive_reselects: int = 5     # R_max [I]
    max_malformed_calls_per_step: int = 3  # M_max [I]
    max_simulator_actions: int = 500       # A_max [P]

    # Success threshold [P]
    success_distance_m: float = 3.0   # meters to goal for success (with stop called)

    # Model parameters
    vlm_model: str = "qwen3.5:4b"
    vlm_thinking: bool = False
    inference_seed: int = 42
    temperature: float = 0.0

    # Ablation toggles (Section 11)
    depth_tool: str = "on"                 # "on", "off", "depth_image"
    recall_tool: bool = True
    history_mode: str = "reasoning"        # "reasoning", "task_progress", "none"
    bev_image: bool = True
    image_history: str = "on_demand"       # "on_demand", "accumulate_all"
    action_mode: str = "pixel"             # "pixel", "waypoint_predictor"
