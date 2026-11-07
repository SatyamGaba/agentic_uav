"""
Step-by-Step Diagnostic Tracking & Evaluation Script for UAV / AgenticNav Pilot Runs.
Evaluates every decision step before advancing to the next step:
- Reasoning Quality & Milestone Deconstruction
- Visual / Depth Perception & Metric Clearance
- Tool Action Parsing & Parameter Normalization
- Physical Movement Consistency & Goal Progress
- Premature Stop & Stalling Detection
"""
import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.data.data_loader import load_episodes
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
from agenticnav_paper_replica.harness.runner import run_episode, EpisodeRunResult
from agenticnav_paper_replica.harness.step_evaluator import StepEvaluator
from agenticnav_paper_replica.traces.trace_logger import TraceLogger
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.run_benchmark import mock_vlm_policy


def main():
    parser = argparse.ArgumentParser(description="Run pilot evaluation with step-by-step diagnostic tracking.")
    parser.add_argument("--episodes-count", type=int, default=1, help="Number of pilot episodes to evaluate (default: 1).")
    parser.add_argument("--mode", type=str, default="verbose", choices=["verbose", "interactive", "autonomous"],
                        help="verbose: print rich telemetry per step; interactive: pause between steps; autonomous: silent run with JSON logging.")
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "ollama", "mock"], help="Inference backend.")
    parser.add_argument("--model", type=str, default="qwen3-vl:8b", help="Model name.")
    parser.add_argument("--out-dir", type=str, default="runs/pilot_step_tracking", help="Output directory for telemetry.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(log_dir=traces_dir)

    config = ReplicaConfig()

    ep_file = "agenticnav_paper_replica/data/episode_list_100.json"
    episodes = load_episodes(ep_file, max_episodes=args.episodes_count)

    print("=" * 80)
    print("      AGENTICNAV STEP-BY-STEP DIAGNOSTIC TRACKING HARNESS")
    print("=" * 80)
    print(f"Tracking Mode:       {args.mode.upper()}")
    print(f"Inference Backend:   {args.backend}")
    print(f"Model:               {args.model}")
    print(f"Episodes to Inspect: {len(episodes)}")
    print(f"Output Telemetry:    {out_dir}")
    print("=" * 80)

    # Initialize VLM Client
    if args.backend == "mock":
        vlm_fn = mock_vlm_policy
        print("Using deterministic test mock policy.")
    else:
        ollama_client = OllamaNavClient(model_name=args.model)
        if ollama_client.is_configured():
            vlm_fn = ollama_client
            print(f"Connected to local Ollama inference server ({args.model}).")
        elif args.backend == "auto":
            print(f"Ollama server not reachable for {args.model}, falling back to mock policy.")
            vlm_fn = mock_vlm_policy
        else:
            raise RuntimeError(f"Ollama inference server is NOT reachable at {ollama_client.base_url} for model '{args.model}'! Terminating to prevent unverified mock fallback.")

    for idx, ep_data in enumerate(episodes):
        ep_id = ep_data["episode_id"]
        inst = ep_data["instruction"]
        start_p = ep_data.get("start_position", [0, 0, 0])
        goal_p = ep_data.get("goal_position", [0, 0, -10])
        sp_dist = ep_data.get("shortest_path_distance", 10.0)

        ep_out_dir = out_dir / ep_id
        ep_out_dir.mkdir(parents=True, exist_ok=True)

        step_evaluator = StepEvaluator(
            mode=args.mode,
            out_dir=ep_out_dir,
            save_intermediate_bev=True,
            interactive_pause=(args.mode == "interactive"),
        )

        print("\n" + "#" * 80)
        print(f" STARTING INSPECTION: EPISODE {idx + 1:02d}/{len(episodes):02d} ({ep_id})")
        print(f" Instruction: \"{inst}\"")
        print(f" Start: {start_p} -> Goal: {goal_p} (Shortest: {sp_dist:.2f}m)")
        print("#" * 80)

        embodiment = MockGroundEmbodiment(max_actions=config.max_simulator_actions)

        # Execute episode with the StepEvaluator actively verifying each step
        res: EpisodeRunResult = run_episode(
            episode_data=ep_data,
            embodiment=embodiment,
            config=config,
            vlm_client_fn=vlm_fn,
            trace_logger=trace_logger,
            step_evaluator=step_evaluator,
        )

        # Save Episode Audit Trace
        step_evaluator.save_episode_audit(ep_id=ep_id)

        # Save Final BEV Map
        if res.final_bev_image is not None:
            final_bev_path = ep_out_dir / f"final_bev_{ep_id}.png"
            Image.fromarray(res.final_bev_image).save(final_bev_path)
            print(f"[EVALUATOR] Final BEV Map saved: {final_bev_path}")

        # Summary of Episode Evaluation
        evals = step_evaluator.evaluations
        total_steps = len(evals)
        consistent_steps = sum(1 for e in evals if e.movement_consistent)
        collisions = sum(1 for e in evals if e.collided)
        stalls = sum(1 for e in evals if e.stalled)
        high_quality_reasoning = sum(1 for e in evals if e.reasoning_quality == "high")
        premature_stop = any(e.premature_stop_detected for e in evals)

        print("\n" + "=" * 80)
        print(f"         EPISODE {ep_id} DIAGNOSTIC SUMMARY")
        print("=" * 80)
        print(f" Outcome Status        : {'SUCCESS' if res.metrics.success else 'FAILURE'} (Category: {res.failure_category})")
        print(f" Total Steps Executed  : {total_steps}")
        print(f" Movement Consistency  : {consistent_steps}/{total_steps} steps consistent ({consistent_steps/max(1, total_steps)*100:.1f}%)")
        print(f" Reasoning Quality     : {high_quality_reasoning}/{total_steps} steps with high-density milestone reasoning")
        print(f" Premature Stop Guard  : {'TRIGGERED ⚠️' if premature_stop else 'PASSED ✓'}")
        print(f" Physical Collisions   : {collisions}")
        print(f" Movement Stalls       : {stalls}")
        print(f" Metric SPL            : {res.metrics.spl:.4f}")
        print(f" Navigation Error (NE) : {res.metrics.navigation_error_m:.2f} meters")
        print(f" Telemetry Saved To    : {ep_out_dir}")
        print("=" * 80)


if __name__ == "__main__":
    main()

