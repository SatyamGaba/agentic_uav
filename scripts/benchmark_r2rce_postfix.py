#!/usr/bin/env python3
"""Post-fix R2R-CE val_unseen benchmark with strict provenance and fail-fast checks.

This script evaluates the active AgenticNav harness on a deterministic subset of
the official R2R-CE val_unseen split. It writes complete evidence under
``audit/phase9_postfix/`` and fails if the run is incomplete or internally
inconsistent.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.data.data_loader import load_episodes
from agenticnav_paper_replica.embodiment.habitat_ground import (
    HabitatGroundEmbodiment,
    HABITAT_AVAILABLE,
    resolve_scene_path,
)
from agenticnav_paper_replica.evaluation.bootstrap import compute_bootstrap_ci
from agenticnav_paper_replica.harness.runner import run_episode
from agenticnav_paper_replica.traces.trace_logger import TraceLogger
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient

DEFAULT_EPISODES = PROJECT_ROOT / "agenticnav_paper_replica/data/r2rce_val_unseen_17.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "audit/phase9_postfix"
FAILURE_TAXONOMY = (
    "safety_deadlock",
    "stop_premature",
    "stop_after_reach",
    "missed_stop",
    "budget_exceeded",
    "protocol_violation",
    "infrastructure_failure",
    "unclassified",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def check_clean_git_tree() -> str:
    """Verifies that the git working tree is clean and returns the current commit hash."""
    rev = git_revision()
    if not rev:
        raise RuntimeError("Not a valid git repository or git rev-parse failed.")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    if status:
        raise RuntimeError(
            f"Git working tree is dirty! Commit all changes before running benchmark.\nDirty files:\n{status}"
        )
    return rev


def wilson_ci(successes: int, total: int, z: float = 1.96) -> Dict[str, Any]:
    if total <= 0:
        return {"mean_pct": 0.0, "ci_95_pct": [0.0, 0.0]}
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denom
    return {
        "mean_pct": round(100.0 * p, 2),
        "ci_95_pct": [
            round(100.0 * max(0.0, centre - half), 2),
            round(100.0 * min(1.0, centre + half), 2),
        ],
    }


def metric_summary(values: List[float], pct: bool = False) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci_95": [0.0, 0.0]}
    ci = compute_bootstrap_ci(arr, n_resamples=10000, random_seed=42)
    scale = 100.0 if pct else 1.0
    return {
        "mean": round(float(ci.mean) * scale, 2 if pct else 4),
        "ci_95": [
            round(float(ci.ci_lower) * scale, 2 if pct else 4),
            round(float(ci.ci_upper) * scale, 2 if pct else 4),
        ],
    }


def empty_failure_counts() -> Dict[str, int]:
    counts = {"success": 0}
    counts.update({key: 0 for key in FAILURE_TAXONOMY})
    return counts


def arrival_verification_count(result: Any) -> int:
    trace = getattr(result, "trace", None)
    if trace is None:
        return 0
    return sum(1 for step in trace.steps if step.get("arrival_verification_triggered"))


def stop_immediately_policy(messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "thought": "Stop-immediately control baseline.",
        "tool_calls": [
            {"name": "stop", "arguments": {"reason": "Control baseline: stop at start."}}
        ],
    }


def run_one(
    episode: Dict[str, Any],
    config: ReplicaConfig,
    vlm_fn: Any,
    trace_logger: TraceLogger,
    model_name: str,
    embodiment: Optional[HabitatGroundEmbodiment] = None,
) -> Dict[str, Any]:
    episode_id = str(episode["episode_id"])
    scene_id = str(episode["scene_id"])
    resolved_scene = resolve_scene_path(scene_id)
    if not os.path.isfile(resolved_scene):
        raise FileNotFoundError(f"Scene '{scene_id}' not found ({resolved_scene}).")

    own_embodiment = False
    if embodiment is None:
        embodiment = HabitatGroundEmbodiment(
            scene_id=resolved_scene,
            max_actions=config.max_simulator_actions,
            gpu_device_id=-1,
        )
        own_embodiment = True
    try:
        result = run_episode(
            episode_data=episode,
            embodiment=embodiment,
            config=config,
            vlm_client_fn=vlm_fn,
            trace_logger=trace_logger,
        )
    finally:
        if own_embodiment:
            embodiment.close()

    metrics = result.metrics
    return {
        "episode_id": episode_id,
        "scene_id": scene_id,
        "instruction": episode.get("instruction", ""),
        "model": model_name,
        "called_stop": result.called_stop,
        "terminated_deadlock": result.terminated_deadlock,
        "terminated_protocol_violation": result.terminated_protocol_violation,
        "total_steps": result.total_steps,
        "arrival_verification_steps": arrival_verification_count(result),
        "failure_category": result.failure_category or "success",
        "metrics": {
            "success": bool(metrics.success),
            "spl": float(metrics.spl),
            "oracle_success": bool(metrics.oracle_success),
            "navigation_error_m": float(metrics.navigation_error_m),
            "trajectory_length_m": float(metrics.trajectory_length_m),
            "simulator_action_count": int(metrics.simulator_action_count),
        },
        "metrics_dataclass": asdict(metrics),
        "shortest_path_distance_m": float(episode.get("shortest_path_distance", 0.0)),
        "goal_radius_m": float(episode.get("goal_radius", 3.0)),
    }


def summarize(
    episodes: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    config: ReplicaConfig,
    model_name: str,
) -> Dict[str, Any]:
    counts = empty_failure_counts()
    for result in results:
        category = result["failure_category"]
        if category not in counts:
            counts[category] = 0
        counts[category] += 1

    sr = [1.0 if r["metrics"]["success"] else 0.0 for r in results]
    spl = [r["metrics"]["spl"] for r in results]
    osr = [1.0 if r["metrics"]["oracle_success"] else 0.0 for r in results]
    ne = [r["metrics"]["navigation_error_m"] for r in results]
    tl = [r["metrics"]["trajectory_length_m"] for r in results]
    actions = [r["metrics"]["simulator_action_count"] for r in results]

    n_success = int(sum(sr))
    n_osr = int(sum(osr))
    return {
        "episode_count": len(results),
        "primary_metrics": {
            "success_rate": {
                **metric_summary(sr, pct=True),
                "wilson_ci_95_pct": wilson_ci(n_success, len(results))["ci_95_pct"],
            },
            "spl": metric_summary(spl, pct=True),
            "oracle_success_rate": {
                **metric_summary(osr, pct=True),
                "wilson_ci_95_pct": wilson_ci(n_osr, len(results))["ci_95_pct"],
            },
            "navigation_error_m": metric_summary(ne),
            "trajectory_length_m": metric_summary(tl),
            "simulator_actions": metric_summary(actions),
        },
        "failure_taxonomy": counts,
        "config": {
            "model": model_name,
            "max_simulator_actions": config.max_simulator_actions,
            "max_decision_steps": config.max_decision_steps,
            "max_evidence_queries_per_step": config.max_evidence_queries_per_step,
            "max_consecutive_reselects": config.max_consecutive_reselects,
            "max_malformed_calls_per_step": config.max_malformed_calls_per_step,
            "success_distance_m": config.success_distance_m,
            "temperature": config.temperature,
            "seed": config.inference_seed,
        },
        "episodes": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-fix R2R-CE benchmark")
    parser.add_argument("--episodes", type=Path, default=DEFAULT_EPISODES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--arm", type=str, choices=["P", "L"], default="L",
                        help="Experimental arm: 'P' (Paper-faithful: N_th=1, no arrival verify, no floor guidance) or 'L' (Local aids: N_th=5, arrival verify, floor guidance).")
    parser.add_argument("--model", type=str, default=None,
                        help="VLM Model ID (or set AGENTICNAV_MODEL_ID env var). Required.")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Allow running with dirty git tree (development/smoke only; strictly forbidden in locked mode).")
    parser.add_argument("--locked", action="store_true",
                        help="Locked benchmark mode: strictly enforces clean git tree and locked exposure controls.")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--max-decision-steps", type=int, default=25)
    parser.add_argument("--no-baseline", "--main-only", action="store_true",
                        help="Skip the stop-immediately control baseline.")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Run only the stop-immediately control; preserve main results.")
    args = parser.parse_args()

    if not HABITAT_AVAILABLE:
        raise RuntimeError("habitat_sim is not installed in the active Python environment.")

    # Model resolution: stop hard-coding; read from args or env var, fail loudly if empty
    if not args.baseline_only:
        model_name = args.model or os.environ.get("AGENTICNAV_MODEL_ID")
        if not model_name:
            raise ValueError(
                "Model ID is not specified! Provide --model <MODEL_ID> or set AGENTICNAV_MODEL_ID environment variable."
            )
    else:
        model_name = "stop_immediately_control"

    is_locked = args.locked or "val_unseen_100" in str(args.episodes) or "locked" in str(args.episodes).lower()
    if is_locked and args.allow_dirty:
        raise RuntimeError("CRITICAL POLICY VIOLATION: --allow-dirty is strictly forbidden in locked evaluation mode!")

    # Enforce clean git working tree unless --allow-dirty is explicitly passed
    if not args.allow_dirty:
        commit_hash = check_clean_git_tree()
    else:
        commit_hash = git_revision() or "dirty_development_tree"

    prompt_file = PROJECT_ROOT / "agenticnav_paper_replica/harness/prompts.py"
    prompt_hash = sha256_file(prompt_file) if prompt_file.exists() else "unknown"
    dataset_hash = sha256_file(Path(args.episodes))

    with open(args.episodes, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    metadata = dataset.get("metadata", {})
    episodes = load_episodes(str(args.episodes), max_episodes=args.max_episodes)
    if not episodes:
        raise ValueError("No episodes selected.")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.baseline_only:
        clear_dirs = ("baseline_stop/traces", "baseline_stop/episodes")
    else:
        clear_dirs = ("episodes", "traces")
        if not args.no_baseline:
            clear_dirs += ("baseline_stop/traces", "baseline_stop/episodes")
    for subdir in clear_dirs:
        target = output_dir / subdir
        target.mkdir(parents=True, exist_ok=True)
        for path in target.iterdir():
            if path.is_file():
                path.unlink()

    config = ReplicaConfig()
    config.max_simulator_actions = args.max_actions
    config.max_decision_steps = args.max_decision_steps
    config.max_evidence_queries_per_step = 4
    config.max_consecutive_reselects = 5
    config.max_malformed_calls_per_step = 3
    config.success_distance_m = 3.0
    config.temperature = 0.0
    config.inference_seed = 42
    config.vlm_model = model_name

    if args.arm == "P":
        config.obstacle_point_threshold = 1
        config.floor_grounding_prompt = False
        config.arrival_verification = False
    else:  # Arm L
        config.obstacle_point_threshold = 5
        config.floor_grounding_prompt = True
        config.arrival_verification = True

    # Sort episodes by scene to eliminate repetitive simulator reload overhead
    episodes = sorted(episodes, key=lambda e: resolve_scene_path(str(e.get("scene_id", ""))))

    if not args.baseline_only:
        ollama_client = OllamaNavClient(
            model_name=model_name,
            temperature=config.temperature,
            seed=config.inference_seed,
            thinking=False,
            num_ctx=config.context_length,
        )
        if not ollama_client.is_configured():
            raise RuntimeError(f"Ollama is not reachable at {ollama_client.base_url}.")
        ollama_url = ollama_client.base_url
    else:
        ollama_client = None
        ollama_url = "none"

    trace_logger = TraceLogger(output_dir / "traces")

    dataset_info = {
        "path": str(args.episodes),
        "split": dataset.get("split", "val_unseen" if "unseen" in str(args.episodes) else ("dev" if "dev" in str(args.episodes) else "unknown")),
        "total_episodes_in_file": len(dataset.get("episodes", [])),
        "selected_episodes": len(episodes),
    }

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": commit_hash,
        "dirty": bool(args.allow_dirty),
        "locked_mode": bool(is_locked),
        "prompt_hash": prompt_hash,
        "dataset_hash": dataset_hash,
        "model_id": model_name,
        "git_revision": commit_hash,
        "python": sys.version,
        "platform": platform.platform(),
        "habitat_sim_available": HABITAT_AVAILABLE,
        "ollama_base_url": ollama_url,
        "dataset": dataset_info,
        "episode_ids": [e["episode_id"] for e in episodes],
        "config": {
            "arm": args.arm,
            "obstacle_point_threshold": config.obstacle_point_threshold,
            "floor_grounding_prompt": config.floor_grounding_prompt,
            "arrival_verification": config.arrival_verification,
            "max_simulator_actions": config.max_simulator_actions,
            "max_decision_steps": config.max_decision_steps,
            "max_evidence_queries_per_step": config.max_evidence_queries_per_step,
            "max_consecutive_reselects": config.max_consecutive_reselects,
            "max_malformed_calls_per_step": config.max_malformed_calls_per_step,
            "success_distance_m": config.success_distance_m,
            "temperature": config.temperature,
            "inference_seed": config.inference_seed,
            "thinking": False,
        },
        "source_hashes": {
            name: sha256_file(PROJECT_ROOT / path)
            for name, path in {
                "executor": "agenticnav_paper_replica/harness/executor.py",
                "step_loop": "agenticnav_paper_replica/harness/step_loop.py",
                "runner": "agenticnav_paper_replica/harness/runner.py",
                "prompts": "agenticnav_paper_replica/harness/prompts.py",
                "failure_classifier": "agenticnav_paper_replica/evaluation/failure_classifier.py",
                "config": "agenticnav_paper_replica/configs/default_config.py",
                "vlm_client": "agenticnav_paper_replica/vlm/ollama_client.py",
            }.items()
        },
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if args.baseline_only:
        baseline_logger = TraceLogger(output_dir / "baseline_stop/traces")
        baseline_results: List[Dict[str, Any]] = []
        cur_scene = None
        cur_emb = None
        # Group episodes by scene to minimize simulator re-initialization (10 loads total instead of 100)
        scene_grouped_episodes = sorted(episodes, key=lambda e: resolve_scene_path(str(e.get("scene_id", ""))))
        ep_result_map = {}
        for idx, episode in enumerate(scene_grouped_episodes, start=1):
            print(f"[baseline {idx}/{len(episodes)}] ep_id={episode['episode_id']} ({episode.get('scene_id')})", flush=True)
            res_scene = resolve_scene_path(str(episode.get("scene_id", "")))
            if res_scene != cur_scene:
                if cur_emb is not None:
                    cur_emb.close()
                cur_scene = res_scene
                cur_emb = HabitatGroundEmbodiment(
                    scene_id=res_scene,
                    max_actions=config.max_simulator_actions,
                    gpu_device_id=-1,
                )
            try:
                result = run_one(
                    episode, config, stop_immediately_policy, baseline_logger, "stop_immediately", embodiment=cur_emb
                )
            except Exception as exc:
                result = {
                    "episode_id": episode["episode_id"],
                    "scene_id": episode.get("scene_id"),
                    "failure_category": "infrastructure_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            ep_result_map[str(episode["episode_id"])] = result
            with open(
                output_dir / "baseline_stop/episodes" / f"{episode['episode_id']}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result, f, indent=2, default=str)
        if cur_emb is not None:
            cur_emb.close()
        # Restore original episode ordering
        baseline_results = [ep_result_map[str(e["episode_id"])] for e in episodes]
        baseline_summary = summarize(episodes, baseline_results, config, "stop_immediately")
        with open(output_dir / "baseline_stop_summary.json", "w", encoding="utf-8") as f:
            json.dump(baseline_summary, f, indent=2, default=str)
        print(f"Baseline SR: {baseline_summary['primary_metrics']['success_rate']}")
        print(f"Baseline NE: {baseline_summary['primary_metrics']['navigation_error_m']}")
        return 0

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    cur_scene = None
    cur_emb = None
    for idx, episode in enumerate(episodes, start=1):
        episode_id = episode["episode_id"]
        print(f"[{idx}/{len(episodes)}] {episode_id} ({episode['scene_id']})", flush=True)
        res_scene = resolve_scene_path(str(episode.get("scene_id", "")))
        if res_scene != cur_scene:
            if cur_emb is not None:
                cur_emb.close()
            cur_scene = res_scene
            cur_emb = HabitatGroundEmbodiment(
                scene_id=res_scene,
                max_actions=config.max_simulator_actions,
                gpu_device_id=-1,
            )
        try:
            result = run_one(episode, config, ollama_client, trace_logger, model_name, embodiment=cur_emb)
        except Exception as exc:
            error_record = {
                "episode_id": episode_id,
                "scene_id": episode.get("scene_id"),
                "failure_category": "infrastructure_failure",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            errors.append(error_record)
            with open(output_dir / "episodes" / f"{episode_id}.json", "w", encoding="utf-8") as f:
                json.dump(error_record, f, indent=2)
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
            continue

        results.append(result)
        with open(output_dir / "episodes" / f"{episode_id}.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(
            f"  success={result['metrics']['success']} "
            f"ne={result['metrics']['navigation_error_m']:.2f}m "
            f"actions={result['metrics']['simulator_action_count']} "
            f"category={result['failure_category']}",
            flush=True,
        )
    if cur_emb is not None:
        cur_emb.close()

    summary = summarize(episodes, results, config, model_name)
    summary["infrastructure_errors"] = errors

    baseline_summary: Optional[Dict[str, Any]] = None
    if not args.no_baseline:
        baseline_logger = TraceLogger(output_dir / "baseline_stop/traces")
        baseline_results = []
        cur_scene = None
        cur_emb = None
        for idx, episode in enumerate(episodes, start=1):
            print(f"[baseline {idx}/{len(episodes)}] {episode['episode_id']}", flush=True)
            res_scene = resolve_scene_path(str(episode.get("scene_id", "")))
            if res_scene != cur_scene:
                if cur_emb is not None:
                    cur_emb.close()
                cur_scene = res_scene
                cur_emb = HabitatGroundEmbodiment(
                    scene_id=res_scene,
                    max_actions=config.max_simulator_actions,
                    gpu_device_id=-1,
                )
            try:
                result = run_one(
                    episode, config, stop_immediately_policy, baseline_logger, "stop_immediately", embodiment=cur_emb
                )
            except Exception as exc:
                baseline_results.append({
                    "episode_id": episode["episode_id"],
                    "scene_id": episode.get("scene_id"),
                    "failure_category": "infrastructure_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
                continue
            baseline_results.append(result)
            with open(
                output_dir / "baseline_stop/episodes" / f"{episode['episode_id']}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(result, f, indent=2, default=str)
        if cur_emb is not None:
            cur_emb.close()
        baseline_summary = summarize(episodes, baseline_results, config, "stop_immediately")

    with open(output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    if baseline_summary is not None:
        with open(output_dir / "baseline_stop_summary.json", "w", encoding="utf-8") as f:
            json.dump(baseline_summary, f, indent=2, default=str)

    trace_path = output_dir / "traces/evaluation_traces.jsonl"
    trace_count = 0
    if trace_path.exists():
        with open(trace_path, "r", encoding="utf-8") as f:
            trace_count = sum(1 for line in f if line.strip())

    gate_checks = {
        "all_episodes_attempted": len(results) + len(errors) == len(episodes),
        "all_episodes_completed": len(errors) == 0 and len(results) == len(episodes),
        "trace_count_matches": trace_count == len(results),
        "unique_episode_ids": len({r["episode_id"] for r in results}) == len(results),
        "taxonomy_valid": all(
            count == 0 or key in {"success", *FAILURE_TAXONOMY}
            for key, count in summary["failure_taxonomy"].items()
        ),
        "deadlocks_have_verification": (not config.arrival_verification) or all(
            r.get("arrival_verification_steps", 0) > 0
            for r in results
            if r["failure_category"] == "safety_deadlock"
        ),
    }
    gate_summary = {
        "status": "PASS" if all(gate_checks.values()) else "FAIL",
        "checks": gate_checks,
        "trace_count": trace_count,
        "result_count": len(results),
        "error_count": len(errors),
    }
    with open(output_dir / "phase9_postfix_gate_summary.json", "w", encoding="utf-8") as f:
        json.dump(gate_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("POST-FIX R2R-CE BENCHMARK")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Episodes: {len(results)}/{len(episodes)} completed")
    print(f"SR: {summary['primary_metrics']['success_rate']}")
    print(f"SPL: {summary['primary_metrics']['spl']}")
    print(f"OSR: {summary['primary_metrics']['oracle_success_rate']}")
    print(f"NE: {summary['primary_metrics']['navigation_error_m']}")
    print(f"Failure taxonomy: {summary['failure_taxonomy']}")
    print(f"Gate: {gate_summary['status']} | {gate_checks}")
    print(f"Summary: {output_dir / 'benchmark_summary.json'}")
    print("=" * 70)

    return 0 if gate_summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
