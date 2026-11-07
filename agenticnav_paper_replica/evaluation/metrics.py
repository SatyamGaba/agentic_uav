"""Standard VLN-CE evaluation metrics: SR, SPL, OSR, TL, NE, nDTW (v7 Specification)."""

from dataclasses import dataclass
import math
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
    ndtw: float = 0.0


def compute_trajectory_length(trajectory: List[Tuple[float, float, float]]) -> float:
    """Sum of Euclidean spatial distances between consecutive trajectory points.
    
    Points are 3D positions (x, y, z) or 2D (x, z). Angular yaw is never included.
    In-place turns (dx=0, dy=0, dz=0) contribute 0.0 to trajectory length.
    """
    if len(trajectory) < 2:
        return 0.0
    arr = np.array(trajectory, dtype=np.float64)
    # If passed 3D points [x, y, z], compute 3D Euclidean displacement; if 2D, compute 2D
    diffs = np.diff(arr, axis=0)
    dists = np.sqrt(np.sum(diffs ** 2, axis=1))
    return float(np.sum(dists))


def compute_dtw(p: np.ndarray, q: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between trajectory p (N, D) and reference q (M, D)."""
    n, m = len(p), len(q)
    if n == 0 or m == 0:
        return float("inf")

    dtw_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(p[i - 1] - q[j - 1]))
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )
    return float(dtw_matrix[n, m])


def compute_ndtw(
    trajectory: List[Tuple[float, float, float]],
    reference_path: Optional[List[List[float]]],
    success_threshold: float = 3.0,
) -> float:
    """Compute normalized Dynamic Time Warping (nDTW) score in [0.0, 1.0].

    Formula (Ilharco et al., 2019):
        nDTW = exp(-DTW(P, Q) / (|Q| * threshold))
    """
    if not trajectory or not reference_path or len(reference_path) == 0:
        return 0.0

    # Project trajectory and reference path to 2D (x, z) ground plane
    # In allocentric trajectory, index 0 is x and index 1 is z.
    p = np.array([[pt[0], pt[1]] for pt in trajectory], dtype=np.float64)
    q_arr = np.array(reference_path, dtype=np.float64)
    # If reference_path is 3D [x, y, z], horizontal coordinates are columns 0 and 2
    if q_arr.shape[1] >= 3:
        q = q_arr[:, [0, 2]]
    else:
        q = q_arr[:, :2]

    dtw_dist = compute_dtw(p, q)
    if np.isinf(dtw_dist):
        return 0.0

    ndtw = math.exp(-dtw_dist / (len(q) * success_threshold))
    return float(np.clip(ndtw, 0.0, 1.0))


def compute_episode_metrics(
    trajectory: List[Tuple[float, float, float]],
    distances_to_goal: List[float],
    shortest_path_length: float,
    simulator_action_count: int,
    called_stop: bool = True,
    success_distance_threshold: float = 3.0,
    reference_path: Optional[List[List[float]]] = None,
) -> EpisodeMetrics:
    """Compute standard VLN-CE evaluation metrics for an episode (Blueprint v7 Section 10.1).

    Args:
        trajectory: List of (x, y, z) agent positions.
        distances_to_goal: Geodesic distances to goal evaluated at each step.
        shortest_path_length: Oracle shortest path distance from start to goal.
        simulator_action_count: Total low-level simulator actions executed.
        called_stop: Whether stop was called by the agent (SR requires stop called [I]).
        success_distance_threshold: Distance threshold for success (default 3.0m [P]).
        reference_path: Ground-truth reference path for nDTW evaluation.

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

    # nDTW
    ndtw = compute_ndtw(trajectory, reference_path, success_distance_threshold)

    return EpisodeMetrics(
        success=success,
        spl=spl,
        oracle_success=oracle_success,
        navigation_error_m=final_error,
        trajectory_length_m=traj_len,
        shortest_path_distance_m=shortest_path_length,
        total_steps=len(trajectory),
        simulator_action_count=simulator_action_count,
        ndtw=ndtw,
    )
