"""Standard VLN-CE evaluation metrics: SR, SPL, OSR, TL, NE (v7 Specification)."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class EpisodeMetrics:
    success: bool
    spl: float
    oracle_success: bool
    navigation_error_m: float
    trajectory_length_m: float
    shortest_path_distance_m: float
    total_steps: int
    simulator_action_count: int


def compute_trajectory_length(trajectory: List[Tuple[float, float, float]]) -> float:
    """Sum of Euclidean distances between consecutive trajectory points."""
    if len(trajectory) < 2:
        return 0.0
    arr = np.array(trajectory, dtype=np.float64)
    diffs = np.diff(arr, axis=0)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.sum(dists))


def compute_episode_metrics(
    trajectory: List[Tuple[float, float, float]],
    distances_to_goal: List[float],
    shortest_path_length: float,
    simulator_action_count: int,
    called_stop: bool = True,
    success_distance_threshold: float = 3.0,
) -> EpisodeMetrics:
    """Compute standard VLN-CE evaluation metrics for an episode (Blueprint v7 Section 10.1).

    Args:
        trajectory: List of (x, y, z) agent positions.
        distances_to_goal: Geodesic distances to goal evaluated at each step.
        shortest_path_length: Oracle shortest path distance from start to goal.
        simulator_action_count: Total low-level simulator actions executed.
        called_stop: Whether stop was called by the agent (SR requires stop called [I]).
        success_distance_threshold: Distance threshold for success (default 3.0m [P]).

    Returns:
        EpisodeMetrics dataclass.
    """
    if not distances_to_goal:
        raise ValueError("distances_to_goal cannot be empty.")

    final_error = float(distances_to_goal[-1])
    min_error = float(np.min(distances_to_goal))

    # Success criteria: stop called and final geodesic distance <= threshold (Section 10.1)
    success = bool(called_stop and (final_error <= success_distance_threshold))
    oracle_success = bool(min_error <= success_distance_threshold)

    # Trajectory length
    traj_len = compute_trajectory_length(trajectory)

    # SPL = S * (L_shortest / max(L_shortest, L_actual))
    if success:
        denom = max(float(shortest_path_length), float(traj_len))
        spl = float(shortest_path_length) / denom if denom > 0 else 1.0
    else:
        spl = 0.0

    return EpisodeMetrics(
        success=success,
        spl=spl,
        oracle_success=oracle_success,
        navigation_error_m=final_error,
        trajectory_length_m=traj_len,
        shortest_path_distance_m=shortest_path_length,
        total_steps=len(trajectory),
        simulator_action_count=simulator_action_count,
    )
