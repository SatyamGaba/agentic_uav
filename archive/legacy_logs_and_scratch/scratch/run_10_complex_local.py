"""
Run 10 Complex Navigation Episodes Locally with Qwen 3.5.
Collects standard R2R / AgenticNav metrics:
- SR (Success Rate)
- SPL (Success weighted by Path Length)
- NE (Navigation Error in meters)
- OSR (Oracle Success Rate)
- TL (Trajectory Length in meters)
- Number of Steps
- Stalls & Collisions

Saves:
- 1 RGB observation frame per test
- Final 3-tier BEV map per test
- Full audit JSON trace per test
- Comprehensive benchmark JSON
"""
import sys
import os
import json
import shutil
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
from agenticnav_paper_replica.harness.runner import run_episode
from agenticnav_paper_replica.harness.step_evaluator import StepEvaluator
from agenticnav_paper_replica.traces.trace_logger import TraceLogger
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient


# 10 Complex episodes from dataset (distances >= 11.10m, diverse layouts & instructions)
COMPLEX_EPISODE_IDS = [
    "r2r_ce_val_unseen_0003",
    "r2r_ce_val_unseen_0004",
    "r2r_ce_val_unseen_0005",
    "r2r_ce_val_unseen_0009",
    "r2r_ce_val_unseen_0010",
    "r2r_ce_val_unseen_0011",
    "r2r_ce_val_unseen_0015",
    "r2r_ce_val_unseen_0016",
    "r2r_ce_val_unseen_0017",
    "r2r_ce_val_unseen_0023",
]

ARTIFACT_DIR = Path(r"C:\Users\Kruti\.gemini\antigravity\brain\845399dd-118d-4b96-a067-6f8e6f44b148")


def main():
    parser = argparse.ArgumentParser(description="Run 10 complex local tests with Qwen 3.5.")
    parser.add_argument("--model", type=str, default="qwen3.5:latest", help="Ollama model name.")
    parser.add_argument("--out-dir", type=str, default="runs/local_10_complex", help="Output directory.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(log_dir=traces_dir)

    # Load dataset
    ep_file = "agenticnav_paper_replica/data/episode_list_100.json"
    with open(ep_file, "r") as f:
        all_episodes = json.load(f)["episodes"]
    
    ep_lookup = {ep["episode_id"]: ep for ep in all_episodes}
    selected_episodes = [ep_lookup[eid] for eid in COMPLEX_EPISODE_IDS if eid in ep_lookup]

    print("=" * 80)
    print("      AGENTICNAV LOCAL BENCHMARK: 10 COMPLEX EPISODES")
    print("=" * 80)
    print(f"Model:               {args.model}")
    print(f"Episodes Count:      {len(selected_episodes)}")
    print(f"Output Directory:    {out_dir}")
    print(f"Artifact Directory:  {ARTIFACT_DIR}")
    print("=" * 80)

    # Connect to Ollama
    client = OllamaNavClient(model_name=args.model)
    if not client.is_configured():
        raise RuntimeError(f"Ollama server not reachable at {client.base_url} for model {args.model}!")
    print(f"[INIT] Successfully connected to Ollama ({args.model}).\n")

    config = ReplicaConfig()
    results = []

    for ep_idx, ep_data in enumerate(selected_episodes):
        ep_id = ep_data["episode_id"]
        inst = ep_data["instruction"]
        sp_dist = ep_data.get("shortest_path_distance", 10.0)
        start_p = ep_data.get("start_position", [0, 0, 0])
        goal_p = ep_data.get("goal_position", [0, 0, -10])

        ep_out_dir = out_dir / ep_id
        ep_out_dir.mkdir(parents=True, exist_ok=True)

        print("-" * 80)
        print(f"[TEST {ep_idx + 1:02d}/10] {ep_id}")
        print(f" Instruction: \"{inst}\"")
        print(f" Shortest Path: {sp_dist:.2f}m | Start: {start_p} -> Goal: {goal_p}")
        print("-" * 80)

        embodiment = MockGroundEmbodiment(max_actions=config.max_simulator_actions)
        
        # Step evaluator in summary mode to keep console focused and fast
        step_evaluator = StepEvaluator(
            mode="summary",
            out_dir=ep_out_dir,
            save_intermediate_bev=False,
            interactive_pause=False,
        )

        # Run episode
        run_res = run_episode(
            episode_data=ep_data,
            embodiment=embodiment,
            config=config,
            vlm_client_fn=client,
            trace_logger=trace_logger,
            step_evaluator=step_evaluator,
        )

        # 1. Save 1 RGB observation frame
        rgb_frame_path = ep_out_dir / "observation_front_rgb.png"
        if hasattr(embodiment, "first_observation") and embodiment.first_observation:
            rgb_arr = embodiment.first_observation.rgb_views.get(0)
            if rgb_arr is not None:
                img = Image.fromarray(rgb_arr.astype(np.uint8))
                img.save(rgb_frame_path)
                # Copy to artifact dir
                art_frame_name = f"complex_ep{ep_idx + 1:02d}_rgb_frame.png"
                shutil.copyfile(rgb_frame_path, ARTIFACT_DIR / art_frame_name)
                print(f" -> Saved RGB Observation Frame: {art_frame_name}")

        # 2. Save Final BEV map
        bev_path = ep_out_dir / "final_bev.png"
        if run_res.final_bev_image is not None:
            bev_img = Image.fromarray(run_res.final_bev_image.astype(np.uint8))
            bev_img.save(bev_path)
            # Copy to artifact dir
            art_bev_name = f"complex_ep{ep_idx + 1:02d}_final_bev.png"
            shutil.copyfile(bev_path, ARTIFACT_DIR / art_bev_name)
            print(f" -> Saved Final BEV Map: {art_bev_name}")

        m = run_res.metrics
        ep_summary = {
            "test_index": ep_idx + 1,
            "episode_id": ep_id,
            "instruction": inst,
            "shortest_path_m": sp_dist,
            "navigation_error_m": round(m.navigation_error_m, 3),
            "success": bool(m.success),
            "spl": round(m.spl, 4),
            "oracle_success": bool(m.oracle_success),
            "trajectory_length_m": round(m.trajectory_length_m, 2),
            "total_steps": run_res.total_steps,
            "simulator_action_count": m.simulator_action_count,
            "failure_category": run_res.failure_category,
            "rgb_frame_file": f"complex_ep{ep_idx + 1:02d}_rgb_frame.png",
            "final_bev_file": f"complex_ep{ep_idx + 1:02d}_final_bev.png",
        }
        results.append(ep_summary)

        status_str = "SUCCESS" if m.success else f"FAIL ({run_res.failure_category})"
        print(f" Outcome: {status_str} | NE: {m.navigation_error_m:.2f}m | SPL: {m.spl:.4f} | Steps: {run_res.total_steps} | Path: {m.trajectory_length_m:.2f}m\n")

        # Save incremental benchmark JSON
        with open(out_dir / "benchmark_summary.json", "w") as f:
            json.dump({
                "model": args.model,
                "episodes_completed": len(results),
                "results": results,
            }, f, indent=2)

    # Compute aggregate metrics
    n = len(results)
    sr = sum(r["success"] for r in results) / n if n > 0 else 0.0
    spl = sum(r["spl"] for r in results) / n if n > 0 else 0.0
    osr = sum(r["oracle_success"] for r in results) / n if n > 0 else 0.0
    mean_ne = sum(r["navigation_error_m"] for r in results) / n if n > 0 else 0.0
    mean_tl = sum(r["trajectory_length_m"] for r in results) / n if n > 0 else 0.0
    mean_steps = sum(r["total_steps"] for r in results) / n if n > 0 else 0.0

    benchmark_final = {
        "benchmark": "AgenticNav 10 Complex Tests",
        "model": args.model,
        "total_episodes": n,
        "aggregate_metrics": {
            "success_rate_sr": round(sr, 4),
            "spl": round(spl, 4),
            "oracle_success_rate_osr": round(osr, 4),
            "mean_navigation_error_m": round(mean_ne, 3),
            "mean_trajectory_length_m": round(mean_tl, 2),
            "mean_steps_per_episode": round(mean_steps, 1),
        },
        "episodes": results,
    }

    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(benchmark_final, f, indent=2)

    print("=" * 80)
    print("                      LOCAL BENCHMARK COMPLETE")
    print("=" * 80)
    print(f" Success Rate (SR)       : {sr * 100:.1f}% ({sum(r['success'] for r in results)}/{n})")
    print(f" SPL                     : {spl:.4f}")
    print(f" Oracle Success Rate     : {osr * 100:.1f}%")
    print(f" Mean Navigation Error   : {mean_ne:.2f} meters")
    print(f" Mean Trajectory Length  : {mean_tl:.2f} meters")
    print(f" Mean Steps per Episode  : {mean_steps:.1f}")
    print(f" Full Results Saved To   : {out_dir / 'benchmark_summary.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()

