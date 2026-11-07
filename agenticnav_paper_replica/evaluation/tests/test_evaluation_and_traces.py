"""Unit tests for metrics, bootstrap CI, failure classification, and trace logger (v7 Specification)."""

import json
from pathlib import Path
import numpy as np
import pytest

from agenticnav_paper_replica.evaluation.metrics import compute_episode_metrics, EpisodeMetrics
from agenticnav_paper_replica.evaluation.bootstrap import (
    compute_bootstrap_ci,
    verify_consistency_criterion,
    compare_paired_ablation,
    paired_bootstrap_test,
    holm_bonferroni_correction,
)
from agenticnav_paper_replica.evaluation.failure_classifier import classify_failure
from agenticnav_paper_replica.traces.trace_logger import TraceLogger, EpisodeTrace


class TestMetrics:
    def test_successful_episode(self):
        # Trajectory of 3 points: length = 1.0 + 1.0 = 2.0m
        traj = [(0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, -2.0)]
        dists = [10.0, 5.0, 1.5]  # final error is 1.5m <= 3.0m
        shortest = 2.0
        m: EpisodeMetrics = compute_episode_metrics(
            trajectory=traj,
            distances_to_goal=dists,
            shortest_path_length=shortest,
            simulator_action_count=8,
            called_stop=True,
        )
        assert m.success is True
        assert m.oracle_success is True
        assert abs(m.navigation_error_m - 1.5) < 1e-4
        assert abs(m.trajectory_length_m - 2.0) < 1e-4
        assert abs(m.spl - 1.0) < 1e-4

    def test_failed_without_stop_called(self):
        # Even if within 3.0m, if stop is not called, it is not a success (Section 10.1)
        traj = [(0.0, 0.0, 0.0), (0.0, 0.0, -2.0)]
        dists = [5.0, 1.5]
        m = compute_episode_metrics(
            trajectory=traj,
            distances_to_goal=dists,
            shortest_path_length=2.0,
            simulator_action_count=8,
            called_stop=False,
        )
        assert m.success is False
        assert m.spl == 0.0
        assert m.oracle_success is True


class TestBootstrapAndStatisticalAcceptance:
    def test_bootstrap_ci_coverage(self):
        rng = np.random.default_rng(42)
        sr = rng.binomial(1, 0.76, size=100)
        res = compute_bootstrap_ci(sr, n_resamples=1000)
        assert res.ci_lower <= res.mean <= res.ci_upper

    def test_consistency_criterion_evaluation(self):
        sr = np.array([1.0] * 76 + [0.0] * 24)
        spl = np.array([0.665] * 100)
        verdict = verify_consistency_criterion(sr, spl, target_sr=76.0, target_spl=66.50, n_resamples=1000)
        assert verdict["overall_replication_pass"] is True
        assert verdict["sr"]["consistent"] is True
        assert verdict["spl"]["consistent"] is True

    def test_paired_ablation_comparison(self):
        full_sr = np.array([1] * 76 + [0] * 24)
        abl_sr = np.array([1] * 50 + [0] * 50)
        full_spl = np.array([0.66] * 100)
        abl_spl = np.array([0.40] * 100)
        comp = compare_paired_ablation(full_sr, abl_sr, full_spl, abl_spl)
        assert comp["sr_diff"] > 0
        assert comp["mcnemar_p_value"] < 0.05
        assert comp["wilcoxon_p_value"] < 0.05
        assert comp["bootstrap_sr"]["significant"] is True

    def test_paired_bootstrap_and_holm_bonferroni(self):
        # Continuous multi-seed scores in [0, 1]
        full_sr = np.array([1.0, 0.67, 1.0, 0.33, 1.0, 1.0, 0.67, 1.0] * 12 + [0.0] * 4)
        abl_sr = np.array([0.33, 0.33, 0.67, 0.0, 0.67, 0.33, 0.33, 0.67] * 12 + [0.0] * 4)
        boot_res = paired_bootstrap_test(full_sr, abl_sr, n_resamples=1000)
        assert boot_res["mean_diff"] > 0
        assert boot_res["p_value"] < 0.05
        assert boot_res["significant"] is True

        p_vals = {
            "wo_depth": 0.001,
            "wo_history": 0.004,
            "wo_bev": 0.02,
        }
        hb = holm_bonferroni_correction(p_vals, alpha=0.05)
        assert hb["wo_depth"]["significant"] is True
        assert hb["wo_history"]["significant"] is True
        assert hb["wo_depth"]["rank"] == 1



class TestFailureClassifierWaterfall:
    def test_1_infrastructure_failure(self):
        cat = classify_failure(
            success=False, oracle_success=False, called_stop=False,
            final_distance_to_goal=5.0, action_budget_exceeded=False,
            deadlock_terminated=False, infrastructure_error=True,
        )
        assert cat == "infrastructure_failure"

    def test_2_stop_premature(self):
        cat = classify_failure(
            success=False, oracle_success=False, called_stop=True,
            final_distance_to_goal=5.5, action_budget_exceeded=False,
            deadlock_terminated=False,
        )
        assert cat == "stop_premature"

    def test_3_stop_after_reach(self):
        # Visited goal (OSR=1) but stopped outside 3.0m or failed to stop in time
        cat = classify_failure(
            success=False, oracle_success=True, called_stop=True,
            final_distance_to_goal=4.2, action_budget_exceeded=False,
            deadlock_terminated=False,
        )
        assert cat == "stop_after_reach"

    def test_4_safety_deadlock(self):
        cat = classify_failure(
            success=False, oracle_success=False, called_stop=False,
            final_distance_to_goal=8.0, action_budget_exceeded=False,
            deadlock_terminated=True,
        )
        assert cat == "safety_deadlock"

    def test_4b_safety_deadlock_near_goal(self):
        # Visited goal (OSR=1) but deadlocked at obstacle without calling stop
        cat = classify_failure(
            success=False, oracle_success=True, called_stop=False,
            final_distance_to_goal=0.18, action_budget_exceeded=False,
            deadlock_terminated=True,
        )
        assert cat == "safety_deadlock"

    def test_5_protocol_violation(self):
        cat = classify_failure(
            success=False, oracle_success=False, called_stop=False,
            final_distance_to_goal=8.0, action_budget_exceeded=False,
            deadlock_terminated=False, protocol_violation=True,
        )
        assert cat == "protocol_violation"

    def test_6_budget_exceeded(self):
        cat = classify_failure(
            success=False, oracle_success=False, called_stop=False,
            final_distance_to_goal=6.0, action_budget_exceeded=True,
            deadlock_terminated=False,
        )
        assert cat == "budget_exceeded"

    def test_7_unclassified(self):
        cat = classify_failure(
            success=False, oracle_success=False, called_stop=False,
            final_distance_to_goal=6.0, action_budget_exceeded=False,
            deadlock_terminated=False,
        )
        assert cat == "unclassified"


class TestFailureTaxonomyConsistency:
    CANONICAL = {
        "safety_deadlock",
        "stop_premature",
        "stop_after_reach",
        "missed_stop",
        "budget_exceeded",
        "protocol_violation",
        "infrastructure_failure",
        "unclassified",
    }

    def test_classifier_outputs_are_canonical(self):
        cases = [
            dict(success=False, oracle_success=False, called_stop=False, final_distance_to_goal=8.0, action_budget_exceeded=False, deadlock_terminated=True),
            dict(success=False, oracle_success=False, called_stop=True, final_distance_to_goal=5.5, action_budget_exceeded=False, deadlock_terminated=False),
            dict(success=False, oracle_success=True, called_stop=True, final_distance_to_goal=4.2, action_budget_exceeded=False, deadlock_terminated=False),
            dict(success=False, oracle_success=True, called_stop=False, final_distance_to_goal=2.0, action_budget_exceeded=False, deadlock_terminated=False),
            dict(success=False, oracle_success=False, called_stop=False, final_distance_to_goal=6.0, action_budget_exceeded=True, deadlock_terminated=False),
            dict(success=False, oracle_success=False, called_stop=False, final_distance_to_goal=8.0, action_budget_exceeded=False, deadlock_terminated=False, protocol_violation=True),
            dict(success=False, oracle_success=False, called_stop=False, final_distance_to_goal=8.0, action_budget_exceeded=False, deadlock_terminated=False, infrastructure_error=True),
            dict(success=False, oracle_success=False, called_stop=False, final_distance_to_goal=6.0, action_budget_exceeded=False, deadlock_terminated=False),
        ]
        for case in cases:
            assert classify_failure(**case) in self.CANONICAL


class TestTraceLogger:
    def test_log_and_export_sft(self, tmp_path):
        logger = TraceLogger(tmp_path / "logs")
        trace = EpisodeTrace(
            episode_id="ep_001",
            scene_id="17DRP5sb8fy",
            instruction="Go to the kitchen counter.",
            shortest_path_distance_m=4.5,
            steps=[
                {
                    "step_idx": 1,
                    "turns": [
                        {"response": {"tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Clear hall"}}]}}
                    ]
                }
            ],
            metrics={"success": True, "spl": 0.90},
        )
        logger.log_episode(trace)
        assert logger.jsonl_path.exists()

        sft_file = tmp_path / "sft.jsonl"
        exported = logger.export_to_sft_format(sft_file, only_successful=True)
        assert exported == 1
        with open(sft_file, "r") as f:
            data = json.loads(f.readline())
            assert data["id"] == "ep_001"
