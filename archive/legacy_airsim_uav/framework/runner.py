"""
Unified VLN Benchmark Runner.
Coordinates adapters, backends, and simulation environments to execute rigorous benchmark runs.
"""
import time
import json
import logging
from typing import List, Dict, Any, Optional
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_env import BaseSimulatorEnv
from framework.types import EpisodeGoal, EpisodeResult, BenchmarkSummary, StepTelemetry
from framework.evaluation.metrics import evaluate_episode, summarize_benchmark, euclidean_distance_3d

logger = logging.getLogger("VLNRunner")


class VLNBenchmarkRunner:
    """
    Decoupled benchmark execution harness.
    Can run any adapter on any simulation environment.
    """
    def __init__(
        self,
        adapter: BaseVLNAdapter,
        env: BaseSimulatorEnv,
        model_name: str = "UnknownModel",
        max_steps_per_episode: int = 25,
        success_radius_m: float = 5.0
    ):
        self.adapter = adapter
        self.env = env
        self.model_name = model_name
        self.max_steps = max_steps_per_episode
        self.success_radius_m = success_radius_m

    def run_episode(self, goal: EpisodeGoal) -> EpisodeResult:
        """Executes a single benchmark episode."""
        t_start = time.time()
        obs = self.env.reset(goal)
        self.adapter.reset(obs)

        trajectory: List[List[float]] = [obs.pose.to_list()]
        telemetry: List[StepTelemetry] = []

        done = False
        step_id = 0

        while not done and step_id < self.max_steps:
            step_id += 1
            t_step0 = time.time()

            # 1. Adapter decision step
            action = self.adapter.step(obs)

            # 2. Simulator execution step
            obs, done, info = self.env.step(action)
            step_dur = time.time() - t_step0

            current_pos = obs.pose.to_list()
            trajectory.append(current_pos)

            dist_to_goal = euclidean_distance_3d(current_pos, goal.goal_pos)
            telemetry.append(StepTelemetry(
                step_id=step_id,
                pose=obs.pose,
                action=action,
                distance_to_goal=dist_to_goal,
                step_duration_s=step_dur,
                depth_at_waypoint=action.depth_m,
                info=info
            ))

        total_dur = time.time() - t_start

        return evaluate_episode(
            episode_id=goal.episode_id,
            instruction=goal.instruction,
            trajectory=trajectory,
            goal_pos=goal.goal_pos,
            gt_trajectory=goal.gt_trajectory,
            success_radius_m=self.success_radius_m,
            coarse_radius_m=goal.coarse_success_distance,
            duration_s=total_dur,
            telemetry=telemetry,
            adapter_name=self.adapter.name,
            model_name=self.model_name,
            metadata={"scene_id": goal.scene_id, "env_name": goal.env_name}
        )

    def run_benchmark(
        self,
        goals: List[EpisodeGoal],
        output_json_path: Optional[str] = None
    ) -> BenchmarkSummary:
        """Executes a batch of episodes and aggregates full benchmark statistics."""
        episodes: List[EpisodeResult] = []

        print(f"\n{'='*75}")
        print(f"  Starting VLN Benchmark: {self.adapter.name} on {self.env.name}")
        print(f"  Model: {self.model_name} | Total Episodes: {len(goals)}")
        print(f"{'='*75}")

        for i, goal in enumerate(goals):
            try:
                ep_res = self.run_episode(goal)
            except Exception as e:
                start_pos = [0.0, 0.0, 5.0]
                init_ne = euclidean_distance_3d(start_pos, goal.goal_pos)
                gt_len = 0.0
                if goal.gt_trajectory and len(goal.gt_trajectory) > 1:
                    gt_len = sum(euclidean_distance_3d(goal.gt_trajectory[j], goal.gt_trajectory[j+1]) for j in range(len(goal.gt_trajectory)-1))

                ep_res = EpisodeResult(
                    episode_id=goal.episode_id,
                    instruction=goal.instruction,
                    success=False,
                    oracle_success=False,
                    navigation_error=init_ne,
                    oracle_navigation_error=init_ne,
                    gt_length=gt_len,
                    pred_length=0.0,
                    spl=0.0,
                    steps_taken=0,
                    duration_s=0.0,
                    success_20m=False,
                    oracle_success_20m=False,
                    spl_20m=0.0,
                    trajectory=[start_pos],
                    telemetry=[]
                )
                print(f"  [{i+1}/{len(goals)}] Ep #{goal.episode_id:03d}: ERROR ({e})", flush=True)

            episodes.append(ep_res)

            status_5m = "SUCC@5m" if ep_res.success else "FAIL@5m"
            status_20m = "SUCC@20m" if ep_res.success_20m else "FAIL@20m"
            print(
                f"  [{i+1}/{len(goals)}] Ep #{ep_res.episode_id:03d}: {status_5m} | {status_20m} | "
                f"NE={ep_res.navigation_error:.2f}m (Min={ep_res.oracle_navigation_error:.2f}m) | "
                f"SPL@5m={ep_res.spl:.4f} | Steps={ep_res.steps_taken} | Dur={ep_res.duration_s:.1f}s",
                flush=True
            )

        summary = summarize_benchmark(
            episodes=episodes,
            adapter_name=self.adapter.name,
            model_name=self.model_name,
            environment_name=self.env.name
        )

        print(f"\n{'-'*75}", flush=True)
        print(f"  DUAL-CRITERIA BENCHMARK SUMMARY RESULTS: {self.adapter.name}", flush=True)
        print(f"{'-'*75}", flush=True)
        print(f"  [Official 20m Radius - OpenFly Baseline Standard]", flush=True)
        print(f"    Success Rate (SR@20m):         {summary.success_rate_20m:.2f}%  (95% CI: [{summary.sr_ci_95_20m[0]:.2f}%, {summary.sr_ci_95_20m[1]:.2f}%])", flush=True)
        print(f"    Oracle Success Rate (OSR@20m):  {summary.oracle_success_rate_20m:.2f}%  (95% CI: [{summary.osr_ci_95_20m[0]:.2f}%, {summary.osr_ci_95_20m[1]:.2f}%])", flush=True)
        print(f"    Mean SPL@20m:                  {summary.mean_spl_20m:.4f}", flush=True)
        print(f"  [Strict 5m Radius - Precision Aerial VLN Standard]", flush=True)
        print(f"    Success Rate (SR@5m):          {summary.success_rate:.2f}%  (95% CI: [{summary.sr_ci_95[0]:.2f}%, {summary.sr_ci_95[1]:.2f}%])", flush=True)
        print(f"    Oracle Success Rate (OSR@5m):   {summary.oracle_success_rate:.2f}%  (95% CI: [{summary.osr_ci_95[0]:.2f}%, {summary.osr_ci_95[1]:.2f}%])", flush=True)
        print(f"    Mean SPL@5m:                   {summary.mean_spl:.4f}", flush=True)
        print(f"  [Trajectory Dynamics]", flush=True)
        print(f"    Mean Navigation Error (NE):    {summary.mean_navigation_error:.2f} m", flush=True)
        print(f"    Mean Oracle Min Distance:      {summary.mean_oracle_ne:.2f} m", flush=True)
        print(f"    Mean Path Length:              {summary.mean_pred_length:.2f} m (GT: {summary.mean_gt_length:.2f} m)", flush=True)
        print(f"    Mean Duration / Episode:       {summary.mean_duration_s:.2f} s", flush=True)
        print(f"{'='*75}\n", flush=True)

        if output_json_path:
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(summary.to_dict(), f, indent=2)
            print(f"Saved benchmark summary to: {output_json_path}", flush=True)

        return summary
