"""
Standardized Aerial Vision-Language Navigation Metrics Evaluator.
Computes SR, OSR, NE, SPL, Trajectory Lengths, and 95% Wilson Confidence Intervals.
Strictly follows official VLN / OpenFly benchmark definitions.
"""
import math
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Union, Optional
from framework.types import EpisodeResult, BenchmarkSummary, StepTelemetry


def euclidean_distance_3d(p1: Union[List[float], Tuple[float, ...]], p2: Union[List[float], Tuple[float, ...]]) -> float:
    """Computes 3D Euclidean distance between two points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def compute_path_length_3d(points: List[List[float]]) -> float:
    """Computes total cumulative Euclidean length along a 3D path."""
    if len(points) < 2:
        return 0.0
    length = 0.0
    for i in range(len(points) - 1):
        length += euclidean_distance_3d(points[i], points[i + 1])
    return length


def compute_wilson_score_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Computes Wilson Score confidence interval for binomial success proportions.
    Returns (lower_bound_pct, upper_bound_pct).
    """
    if total == 0:
        return 0.0, 0.0
    z = 1.95996  # 95% two-sided normal quantile
    p = successes / float(total)
    denominator = 1.0 + (z * z) / float(total)
    center = (p + (z * z) / (2.0 * float(total))) / denominator
    spread = (z * math.sqrt((p * (1.0 - p) / float(total)) + ((z * z) / (4.0 * (float(total) ** 2))))) / denominator
    lower = max(0.0, (center - spread) * 100.0)
    upper = min(100.0, (center + spread) * 100.0)
    return round(lower, 2), round(upper, 2)


def evaluate_episode(
    episode_id: Union[int, str],
    instruction: str,
    trajectory: List[List[float]],
    goal_pos: List[float],
    gt_trajectory: List[List[float]],
    success_radius_m: float = 5.0,
    coarse_radius_m: float = 20.0,
    duration_s: float = 0.0,
    telemetry: Optional[List[StepTelemetry]] = None,
    adapter_name: str = "Adapter",
    model_name: str = "Model",
    metadata: Optional[Dict[str, Any]] = None
) -> EpisodeResult:
    """
    Computes all standard metrics for a completed navigation episode across both 5m and 20m radii.
    """
    if not trajectory:
        trajectory = [[0.0, 0.0, 0.0]]

    final_pos = trajectory[-1]

    # 1. Navigation Error (NE): final distance to goal
    final_ne = euclidean_distance_3d(final_pos, goal_pos)

    # 2. Oracle Navigation Error: minimum distance achieved at any point
    all_dists = [euclidean_distance_3d(pt, goal_pos) for pt in trajectory]
    oracle_ne = min(all_dists)

    # 3. Path Lengths
    pred_len = compute_path_length_3d(trajectory)
    gt_len = compute_path_length_3d(gt_trajectory) if gt_trajectory else euclidean_distance_3d(trajectory[0], goal_pos)

    # 4. Strict Metrics (@ 5m)
    is_success_5m = bool(final_ne <= success_radius_m)
    is_oracle_5m = bool(oracle_ne <= success_radius_m)
    spl_5m = float(gt_len / max(gt_len, pred_len)) if (is_success_5m and max(gt_len, pred_len) > 0.0) else 0.0

    # 5. Benchmark Metrics (@ 20m)
    is_success_20m = bool(final_ne <= coarse_radius_m)
    is_oracle_20m = bool(oracle_ne <= coarse_radius_m)
    spl_20m = float(gt_len / max(gt_len, pred_len)) if (is_success_20m and max(gt_len, pred_len) > 0.0) else 0.0

    return EpisodeResult(
        episode_id=episode_id,
        instruction=instruction,
        success=is_success_5m,
        oracle_success=is_oracle_5m,
        navigation_error=round(final_ne, 2),
        oracle_navigation_error=round(oracle_ne, 2),
        gt_length=round(gt_len, 2),
        pred_length=round(pred_len, 2),
        spl=round(spl_5m, 4),
        steps_taken=len(trajectory) - 1,
        duration_s=round(duration_s, 2),
        success_20m=is_success_20m,
        oracle_success_20m=is_oracle_20m,
        spl_20m=round(spl_20m, 4),
        trajectory=trajectory,
        telemetry=telemetry or [],
        adapter_name=adapter_name,
        model_name=model_name,
        metadata=metadata or {}
    )


def summarize_benchmark(
    episodes: List[EpisodeResult],
    adapter_name: str,
    model_name: str,
    environment_name: str
) -> BenchmarkSummary:
    """
    Aggregates individual episode results into a comprehensive dual-threshold BenchmarkSummary.
    """
    total = len(episodes)
    if total == 0:
        return BenchmarkSummary(
            adapter_name=adapter_name,
            model_name=model_name,
            environment_name=environment_name,
            total_episodes=0,
            success_count=0,
            oracle_success_count=0,
            success_rate=0.0,
            sr_ci_95=(0.0, 0.0),
            oracle_success_rate=0.0,
            osr_ci_95=(0.0, 0.0),
            mean_spl=0.0,
            success_count_20m=0,
            oracle_success_count_20m=0,
            success_rate_20m=0.0,
            sr_ci_95_20m=(0.0, 0.0),
            oracle_success_rate_20m=0.0,
            osr_ci_95_20m=(0.0, 0.0),
            mean_spl_20m=0.0,
            mean_navigation_error=0.0,
            mean_oracle_ne=0.0,
            mean_pred_length=0.0,
            mean_gt_length=0.0,
            mean_duration_s=0.0,
            mean_steps=0.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            episodes=[]
        )

    # 5m Metrics
    success_cnt_5m = sum(1 for ep in episodes if ep.success)
    oracle_cnt_5m = sum(1 for ep in episodes if ep.oracle_success)
    sr_pct_5m = (success_cnt_5m / float(total)) * 100.0
    osr_pct_5m = (oracle_cnt_5m / float(total)) * 100.0
    sr_ci_5m = compute_wilson_score_interval(success_cnt_5m, total)
    osr_ci_5m = compute_wilson_score_interval(oracle_cnt_5m, total)
    mean_spl_5m = sum(ep.spl for ep in episodes) / float(total)

    # 20m Metrics
    success_cnt_20m = sum(1 for ep in episodes if ep.success_20m)
    oracle_cnt_20m = sum(1 for ep in episodes if ep.oracle_success_20m)
    sr_pct_20m = (success_cnt_20m / float(total)) * 100.0
    osr_pct_20m = (oracle_cnt_20m / float(total)) * 100.0
    sr_ci_20m = compute_wilson_score_interval(success_cnt_20m, total)
    osr_ci_20m = compute_wilson_score_interval(oracle_cnt_20m, total)
    mean_spl_20m = sum(ep.spl_20m for ep in episodes) / float(total)

    mean_ne = sum(ep.navigation_error for ep in episodes) / float(total)
    mean_oracle_ne = sum(ep.oracle_navigation_error for ep in episodes) / float(total)
    mean_pred_len = sum(ep.pred_length for ep in episodes) / float(total)
    mean_gt_len = sum(ep.gt_length for ep in episodes) / float(total)
    mean_dur = sum(ep.duration_s for ep in episodes) / float(total)
    mean_steps = sum(ep.steps_taken for ep in episodes) / float(total)

    return BenchmarkSummary(
        adapter_name=adapter_name,
        model_name=model_name,
        environment_name=environment_name,
        total_episodes=total,
        success_count=success_cnt_5m,
        oracle_success_count=oracle_cnt_5m,
        success_rate=round(sr_pct_5m, 2),
        sr_ci_95=sr_ci_5m,
        oracle_success_rate=round(osr_pct_5m, 2),
        osr_ci_95=osr_ci_5m,
        mean_spl=round(mean_spl_5m, 4),
        success_count_20m=success_cnt_20m,
        oracle_success_count_20m=oracle_cnt_20m,
        success_rate_20m=round(sr_pct_20m, 2),
        sr_ci_95_20m=sr_ci_20m,
        oracle_success_rate_20m=round(osr_pct_20m, 2),
        osr_ci_95_20m=osr_ci_20m,
        mean_spl_20m=round(mean_spl_20m, 4),
        mean_navigation_error=round(mean_ne, 2),
        mean_oracle_ne=round(mean_oracle_ne, 2),
        mean_pred_length=round(mean_pred_len, 2),
        mean_gt_length=round(mean_gt_len, 2),
        mean_duration_s=round(mean_dur, 2),
        mean_steps=round(mean_steps, 1),
        timestamp=datetime.now(timezone.utc).isoformat(),
        episodes=episodes
    )
