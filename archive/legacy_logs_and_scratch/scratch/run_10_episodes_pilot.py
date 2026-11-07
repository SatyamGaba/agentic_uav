"""
10-Episode Pilot Evaluation with Deep Telemetry and BEV Map Exports.
Logs step-by-step movement, reasoning, and saves the final BEV map for each episode.
"""
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List
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
from dataclasses import is_dataclass, asdict

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run 10 episodes evaluation with BEV map exports.")
    parser.add_argument("--episodes-count", type=int, default=10)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "ollama", "mock"])
    parser.add_argument("--model", type=str, default="qwen3-vl:8b")
    parser.add_argument("--out-dir", type=str, default="runs/eval_10ep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bev_dir = out_dir / "bev_maps"
    bev_dir.mkdir(parents=True, exist_ok=True)
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(log_dir=traces_dir)

    config = ReplicaConfig()

    # Load 10 episodes from benchmark dataset
    ep_file = "agenticnav_paper_replica/data/episode_list_100.json"
    episodes = load_episodes(ep_file, max_episodes=args.episodes_count)

    print("=" * 80)
    print(f"       AGENTICNAV 10-EPISODE BENCHMARK RUN & BEV EXPORT")
    print("=" * 80)
    print(f"Backend:     {args.backend}")
    print(f"Model:       {args.model}")
    print(f"Num Episodes:{len(episodes)}")
    print(f"Output Dir:  {out_dir}")
    print("=" * 80)

    # Initialize client
    if args.backend == "mock":
        vlm_fn = mock_vlm_policy
        print("Using deterministic test policy.")
    else:
        ollama_client = OllamaNavClient(model_name=args.model)
        if ollama_client.is_configured():
            vlm_fn = ollama_client
            print(f"Connected to local Ollama inference server ({args.model}).")
        else:
            print("Ollama server not reachable at localhost:11434, falling back to mock policy.")
            vlm_fn = mock_vlm_policy

    results: List[Dict[str, Any]] = []

    for idx, ep_data in enumerate(episodes):
        ep_id = ep_data["episode_id"]
        inst = ep_data["instruction"]
        start_p = ep_data.get("start_position", [0, 0, 0])
        goal_p = ep_data.get("goal_position", [0, 0, -10])
        sp_dist = ep_data.get("shortest_path_distance", 10.0)

        print("\n" + "=" * 80)
        print(f" [EPISODE {idx + 1:02d}/{len(episodes):02d}] {ep_id}")
        print(f" Instruction:   \"{inst}\"")
        print(f" Start / Goal:   Start {start_p} -> Goal {goal_p} (Shortest: {sp_dist:.2f}m)")
        print("-" * 80)

        embodiment = MockGroundEmbodiment(max_actions=config.max_simulator_actions)

        step_evaluator = StepEvaluator(
            mode="summary",
            out_dir=out_dir / "step_diagnostics" / ep_id,
            save_intermediate_bev=True,
            interactive_pause=False,
        )

        # Execute episode
        res: EpisodeRunResult = run_episode(
            episode_data=ep_data,
            embodiment=embodiment,
            config=config,
            vlm_client_fn=vlm_fn,
            trace_logger=trace_logger,
            step_evaluator=step_evaluator,
        )

        step_evaluator.save_episode_audit(ep_id=ep_id)

        trace = res.trace

        # Save Final BEV Map
        bev_path = bev_dir / f"bev_map_{ep_id}.png"
        if res.final_bev_image is not None:
            Image.fromarray(res.final_bev_image).save(bev_path)
            print(f"  -> Final BEV Map saved: {bev_path}")

        # Print step-by-step telemetry
        print(f"\n  --- Execution Steps & Reasoning ---")
        for step_obj in trace.steps:
            s = asdict(step_obj) if is_dataclass(step_obj) else (step_obj if isinstance(step_obj, dict) else vars(step_obj))
            step_idx = s.get("step_idx", 0)
            node_idx = s.get("node_idx", 0)
            pose = s.get("agent_pose", {"x": 0, "z": 0, "yaw_rad": 0})
            geod = s.get("geodesic_distance_to_goal_m", 0.0)
            act_type = s.get("action_type", "")
            act_payload = s.get("action_payload", {})
            turns = s.get("turns", [])

            print(f"  [Step {step_idx:02d} @ Node {node_idx:02d}] Pose: (x={pose['x']:+5.2f}m, z={pose['z']:+5.2f}m, yaw={np.degrees(pose['yaw_rad']):+6.1f} deg) | Goal Dist: {geod:5.2f}m")
            for t_i, turn_obj in enumerate(turns):
                t = asdict(turn_obj) if is_dataclass(turn_obj) else (turn_obj if isinstance(turn_obj, dict) else vars(turn_obj))
                for tc in t.get("tool_calls", []):
                    print(f"     Turn {t_i}: Tool -> {tc.get('name')}({tc.get('arguments')})")

            if act_type == "move_to":
                print(f"     -> ACTION: move_to | Moved {act_payload.get('moved_m', 0.0):.2f}m @ {act_payload.get('theta_deg', 0.0):+.1f} deg | Collided: {act_payload.get('collided', False)}")
                print(f"        Reason: \"{act_payload.get('reason', '')}\"")
            elif act_type == "stop":
                print(f"     -> ACTION: stop | Reason: \"{act_payload.get('reason', '')}\"")
            else:
                print(f"     -> TERMINATION: {act_type} | Payload: {act_payload}")

        ep_summary = {
            "episode_id": ep_id,
            "success": res.metrics.success,
            "spl": res.metrics.spl,
            "navigation_error_m": res.metrics.navigation_error_m,
            "oracle_success": res.metrics.oracle_success,
            "trajectory_length_m": res.metrics.trajectory_length_m,
            "shortest_path_distance_m": res.metrics.shortest_path_distance_m,
            "failure_category": res.failure_category,
            "total_steps": res.total_steps,
            "bev_map_file": str(bev_path),
        }
        results.append(ep_summary)

        print(f"\n  [Episode Outcome] Success: {res.metrics.success} | SPL: {res.metrics.spl:.4f} | NE: {res.metrics.navigation_error_m:.2f}m | Cat: {res.failure_category}")

    # Compute aggregate metrics across 10 episodes
    num_ep = len(results)
    sr = float(np.mean([1.0 if r["success"] else 0.0 for r in results]) * 100.0)
    spl = float(np.mean([r["spl"] for r in results]) * 100.0)
    osr = float(np.mean([1.0 if r["oracle_success"] else 0.0 for r in results]) * 100.0)
    mean_ne = float(np.mean([r["navigation_error_m"] for r in results]))

    failure_dist: Dict[str, int] = {}
    for r in results:
        cat = r["failure_category"] or "success"
        failure_dist[cat] = failure_dist.get(cat, 0) + 1

    summary_data = {
        "num_episodes": num_ep,
        "metrics": {
            "sr": {"mean": sr, "target": 76.0},
            "spl": {"mean": spl, "target": 66.5},
            "osr": {"mean": osr, "target": 78.0},
            "ne_m": {"mean": mean_ne, "target": 3.48},
        },
        "failure_distribution": failure_dist,
        "episodes": results,
    }

    summary_file = out_dir / "summary_metrics.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 80)
    print("                 10-EPISODE BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Total Episodes Evaluated : {num_ep}")
    print(f"Success Rate (SR)        : {sr:5.1f}%   (Target: 76.0%)")
    print(f"SPL                      : {spl:5.2f}%   (Target: 66.50%)")
    print(f"Oracle Success Rate (OSR): {osr:5.1f}%   (Target: 78.0%)")
    print(f"Mean Navigation Error    : {mean_ne:5.2f}m  (Target: 3.48m)")
    print(f"Failure Distribution     : {failure_dist}")
    print(f"Summary metrics saved    : {summary_file}")
    print(f"BEV maps directory       : {bev_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

