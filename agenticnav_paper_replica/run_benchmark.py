"""Main benchmark execution harness for AgenticNav replication (v7 Specification).

Runs Full System or Ablation arms across K repetitions, computes 10,000-resample
non-parametric bootstrap confidence intervals, and verifies Consistency Rule (a).
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Any, Callable, Dict, List, Optional
import numpy as np

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.data.data_loader import load_episodes
from agenticnav_paper_replica.embodiment.base import EmbodimentBase
from agenticnav_paper_replica.embodiment.habitat_ground import (
    HabitatGroundEmbodiment,
    HABITAT_AVAILABLE,
    resolve_scene_path,
)
from agenticnav_paper_replica.evaluation.bootstrap import (
    verify_consistency_criterion,
    compute_bootstrap_ci,
)
from agenticnav_paper_replica.harness.runner import run_episode, EpisodeRunResult
from agenticnav_paper_replica.traces.trace_logger import TraceLogger
from agenticnav_paper_replica.vlm.gemini_client import GeminiNavClient
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient


def create_ablation_config(ablation_name: str) -> ReplicaConfig:
    """Configures ReplicaConfig for the requested experimental arm."""
    cfg = ReplicaConfig()

    if ablation_name == "full":
        return cfg
    elif ablation_name == "wo_depth":
        cfg.depth_tool = "off"
    elif ablation_name == "wo_recall":
        cfg.recall_tool = False
    elif ablation_name == "wo_history":
        cfg.history_mode = "none"
    elif ablation_name == "wo_bev":
        cfg.bev_image = False
    elif ablation_name == "depth_image":
        cfg.depth_tool = "depth_image"
    else:
        raise ValueError(f"Unknown ablation arm: {ablation_name}")

    return cfg


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


def main():
    parser = argparse.ArgumentParser(description="AgenticNav Paper Replica Benchmark Runner")
    parser.add_argument("--episodes", type=str, default="agenticnav_paper_replica/data/r2rce_val_unseen_17.json",
                        help="Path to episode JSON file.")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes to run (e.g. 5 for dev smoke-test).")
    parser.add_argument("--repetitions", "-K", type=int, default=3,
                        help="Number of repeated evaluations K per episode (default 3).")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "wo_depth", "wo_recall", "wo_history", "wo_bev", "depth_image"],
                        help="Experimental arm.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save traces and summary metrics.")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "gemini", "ollama"],
                        help="VLM backend provider (default 'auto' prefers Gemini, then Ollama; failure is fatal).")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Model name (e.g. 'gemini-2.5-flash' for Gemini, 'qwen3-vl:8b' for Ollama).")
    parser.add_argument("--resume", action="store_true", default=True,
                        help="Resume from existing checkpoint in output-dir if present.")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Do not resume; overwrite existing output directory.")
    args = parser.parse_args()

    # 1. Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"runs/{args.ablation}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(log_dir=out_dir / "traces")

    # 2. Load episodes (fail fast: never silently substitute synthetic data)
    if not Path(args.episodes).exists():
        raise FileNotFoundError(f"Episode file not found: {args.episodes}")
    episodes = load_episodes(args.episodes, max_episodes=args.max_episodes)

    num_episodes = len(episodes)
    print(f"Loaded {num_episodes} episodes for arm '{args.ablation}' with K={args.repetitions} repetitions.")

    # 3. Setup configuration and client
    config = create_ablation_config(args.ablation)

    ollama_model = args.model if args.model != "gemini-2.5-flash" else config.vlm_model

    if args.backend == "ollama":
        ollama_client = OllamaNavClient(model_name=ollama_model)
        if not ollama_client.is_configured():
            raise RuntimeError(f"Ollama server is not reachable at {ollama_client.base_url}.")
        vlm_fn = ollama_client
        print(f"Connected to local Ollama inference server ({ollama_model}).")
    elif args.backend == "gemini":
        gemini_client = GeminiNavClient(model_name=args.model)
        if not gemini_client.is_configured():
            raise RuntimeError("GEMINI_API_KEY environment variable is required for --backend gemini.")
        vlm_fn = gemini_client
        print(f"Connected to Gemini API ({args.model}).")
    else:
        # Auto detection: Gemini -> Ollama -> fatal. Never fall back to mock.
        gemini_client = GeminiNavClient(model_name=args.model)
        if gemini_client.is_configured():
            vlm_fn = gemini_client
            print(f"[AUTO] Connected to Gemini API ({args.model}).")
        else:
            ollama_client = OllamaNavClient(model_name=ollama_model)
            if not ollama_client.is_configured():
                raise RuntimeError(
                    "[AUTO] No VLM backend available: Gemini credentials missing and "
                    f"Ollama is not reachable at {ollama_client.base_url}."
                )
            vlm_fn = ollama_client
            print(f"[AUTO] Connected to local Ollama ({ollama_client.model_name}).")

    # 4. Run benchmark with persistent checkpointing and live progress tracking
    checkpoint_file = out_dir / "checkpoint.json"
    live_progress_file = out_dir / "live_progress.json"

    # Matrix of results: (K, N)
    sr_matrix = np.zeros((args.repetitions, num_episodes), dtype=np.float64)
    spl_matrix = np.zeros((args.repetitions, num_episodes), dtype=np.float64)
    ne_matrix = np.zeros((args.repetitions, num_episodes), dtype=np.float64)
    osr_matrix = np.zeros((args.repetitions, num_episodes), dtype=np.float64)
    failure_counts: Dict[str, int] = {"success": 0}
    failure_counts.update({k: 0 for k in FAILURE_TAXONOMY})
    completed_keys = set()

    # Resume from checkpoint if available
    if args.resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
            completed_keys = set(ckpt.get("completed_keys", []))
            sr_matrix = np.array(ckpt.get("sr_matrix", sr_matrix))
            spl_matrix = np.array(ckpt.get("spl_matrix", spl_matrix))
            ne_matrix = np.array(ckpt.get("ne_matrix", ne_matrix))
            osr_matrix = np.array(ckpt.get("osr_matrix", osr_matrix))
            failure_counts = ckpt.get("failure_counts", {})
            for taxonomy_key in ("success",) + FAILURE_TAXONOMY:
                failure_counts.setdefault(taxonomy_key, 0)
            print(f"[RESUME] Found existing checkpoint with {len(completed_keys)} completed runs. Resuming...", flush=True)
        except Exception as e:
            print(f"[RESUME WARNING] Failed to load checkpoint: {e}. Starting fresh.", flush=True)

    start_time = datetime.now()
    total_runs = args.repetitions * num_episodes

    for k in range(args.repetitions):
        print(f"\n--- Starting Repetition {k+1}/{args.repetitions} ---", flush=True)
        for ep_idx, ep_data in enumerate(episodes):
            ep_id = ep_data.get("episode_id", f"ep_{ep_idx}")
            run_key = f"k{k}_ep{ep_id}"

            if run_key in completed_keys:
                print(f"[SKIP] Rep {k+1} Ep {ep_idx+1:03d}/{num_episodes} ({ep_id}) already completed.", flush=True)
                continue

            # Instantiate embodiment (fail fast: never silently substitute the mock embodiment)
            if not HABITAT_AVAILABLE:
                raise RuntimeError("habitat_sim is not installed in the active Python environment.")
            scene_id = ep_data.get("scene_id", "")
            resolved_scene = resolve_scene_path(scene_id)
            if not os.path.isfile(resolved_scene):
                raise FileNotFoundError(f"Scene '{scene_id}' .glb not found (resolved: {resolved_scene}).")
            embodiment: EmbodimentBase = HabitatGroundEmbodiment(
                scene_id=resolved_scene,
                max_actions=config.max_simulator_actions,
            )

            try:
                res: EpisodeRunResult = run_episode(
                    episode_data=ep_data,
                    embodiment=embodiment,
                    config=config,
                    vlm_client_fn=vlm_fn,
                    trace_logger=trace_logger,
                )

                sr_val = 1.0 if res.metrics.success else 0.0
                spl_val = res.metrics.spl
                ne_val = res.metrics.navigation_error_m
                osr_val = 1.0 if res.metrics.oracle_success else 0.0
                f_cat = res.failure_category or "success"

            except Exception as e:
                import traceback
                print(f"[ERROR in Ep {ep_id}]: {e}\n{traceback.format_exc()}", flush=True)
                sr_val, spl_val, ne_val, osr_val = 0.0, 0.0, 99.0, 0.0
                f_cat = "infrastructure_failure"

            finally:
                embodiment.close()

            sr_matrix[k, ep_idx] = sr_val
            spl_matrix[k, ep_idx] = spl_val
            ne_matrix[k, ep_idx] = ne_val
            osr_matrix[k, ep_idx] = osr_val
            if f_cat not in failure_counts:
                failure_counts[f_cat] = 0
            failure_counts[f_cat] += 1
            completed_keys.add(run_key)

            # Atomic checkpoint save
            ckpt_data = {
                "completed_keys": list(completed_keys),
                "sr_matrix": sr_matrix.tolist(),
                "spl_matrix": spl_matrix.tolist(),
                "ne_matrix": ne_matrix.tolist(),
                "osr_matrix": osr_matrix.tolist(),
                "failure_counts": failure_counts,
                "timestamp": datetime.now().isoformat(),
            }
            tmp_ckpt = checkpoint_file.with_suffix(".tmp")
            with open(tmp_ckpt, "w", encoding="utf-8") as f:
                json.dump(ckpt_data, f, indent=2)
            tmp_ckpt.replace(checkpoint_file)

            # Compute running telemetry
            done_count = len(completed_keys)
            elapsed_sec = (datetime.now() - start_time).total_seconds()
            eta_sec = (total_runs - done_count) * (elapsed_sec / max(1, done_count))
            eta_min = eta_sec / 60.0

            # Print structured progress line for log tailing / monitor
            print(
                f"[PROGRESS] k={k+1}/{args.repetitions} ep={ep_idx+1:03d}/{num_episodes} "
                f"({done_count}/{total_runs}, {100.0*done_count/total_runs:.1f}%) | "
                f"SR={sr_val > 0} | SPL={spl_val:.3f} | NE={ne_val:.2f}m | Cat={f_cat} | ETA={eta_min:.1f}m",
                flush=True
            )

            # Save live progress JSON atomically
            prog_data = {
                "timestamp": datetime.now().isoformat(),
                "ablation_arm": args.ablation,
                "current_k": k + 1,
                "repetitions_K": args.repetitions,
                "current_ep_idx": ep_idx + 1,
                "total_episodes": num_episodes,
                "completed_runs": done_count,
                "total_runs": total_runs,
                "percent_complete": round(100.0 * done_count / total_runs, 1),
                "running_sr_percent": round(float(np.mean(sr_matrix[:k+1, :ep_idx+1] > 0)) * 100.0, 1),
                "running_spl_percent": round(float(np.mean(spl_matrix[:k+1, :ep_idx+1])) * 100.0, 1),
                "elapsed_minutes": round(elapsed_sec / 60.0, 1),
                "eta_minutes": round(eta_min, 1),
                "last_run": {
                    "key": run_key,
                    "episode_id": ep_id,
                    "success": bool(sr_val > 0),
                    "spl": round(float(spl_val), 4),
                    "ne_m": round(float(ne_val), 2),
                    "failure_category": f_cat,
                }
            }
            tmp_prog = live_progress_file.with_suffix(".tmp")
            with open(tmp_prog, "w", encoding="utf-8") as f:
                json.dump(prog_data, f, indent=2)
            tmp_prog.replace(live_progress_file)

    # 5. Aggregate metrics across K repetitions
    # Per-episode means across K runs
    sr_per_episode = np.mean(sr_matrix, axis=0)
    spl_per_episode = np.mean(spl_matrix, axis=0)
    ne_per_episode = np.mean(ne_matrix, axis=0)
    osr_per_episode = np.mean(osr_matrix, axis=0)

    # 6. Bootstrap confidence intervals and Consistency Criterion Rule (a)
    # Compare against the paper's open-weight Qwen-family row (not the Gemini anchor).
    consistency_report = verify_consistency_criterion(
        sr_per_episode=sr_per_episode,
        spl_per_episode=spl_per_episode,
        target_sr=48.0,
        target_spl=33.57,
        n_resamples=10000,
    )

    ci_osr = compute_bootstrap_ci(osr_per_episode * 100.0)
    ci_ne = compute_bootstrap_ci(ne_per_episode)

    summary = {
        "timestamp": timestamp,
        "ablation_arm": args.ablation,
        "repetitions_K": args.repetitions,
        "num_episodes": num_episodes,
        "consistency_rule_a": consistency_report,
        "metrics": {
            "sr": consistency_report["sr"],
            "spl": consistency_report["spl"],
            "osr": {
                "mean": round(ci_osr.mean, 2),
                "ci_95": [round(ci_osr.ci_lower, 2), round(ci_osr.ci_upper, 2)],
                "target": 72.0,
                "target_source": "paper open-weight Qwen-family row",
            },
            "ne_m": {
                "mean": round(ci_ne.mean, 2),
                "ci_95": [round(ci_ne.ci_lower, 2), round(ci_ne.ci_upper, 2)],
                "target": 5.65,
                "target_source": "paper open-weight Qwen-family row",
            },
        },
        "failure_distribution": failure_counts,
    }

    summary_file = out_dir / "summary_metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("AgenticNav Paper Replication Benchmark Results")
    print("=" * 60)
    print(f"Ablation Arm:     {args.ablation}")
    print(f"Repetitions K:    {args.repetitions}")
    print(f"Total Episodes:   {num_episodes}")
    print(f"SR:               {consistency_report['sr']['mean']:.2f}% (Open-weight target: 48.0%, CI95: {consistency_report['sr']['ci_95']})")
    print(f"SPL:              {consistency_report['spl']['mean']:.2f}% (Open-weight target: 33.57%, CI95: {consistency_report['spl']['ci_95']})")
    print(f"OSR:              {ci_osr.mean:.2f}% (Open-weight target: 72.0%, CI95: [{ci_osr.ci_lower:.2f}, {ci_osr.ci_upper:.2f}])")
    print(f"NE:               {ci_ne.mean:.2f}m (Open-weight target: 5.65m, CI95: [{ci_ne.ci_lower:.2f}, {ci_ne.ci_upper:.2f}])")
    print(f"Consistency Pass: {consistency_report['overall_replication_pass']}")
    print(f"Failure Taxonomy: {failure_counts}")
    print(f"Summary JSON:     {summary_file}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
