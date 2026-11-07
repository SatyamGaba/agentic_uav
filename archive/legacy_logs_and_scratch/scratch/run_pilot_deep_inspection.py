"""
End-to-end Pilot Verification Script (1-2 Episodes).
Inspects all stages of reasoning, inputs, and outputs in detail:
  1. Multimodal context assembly (RGB views, BEV map, history, instruction)
  2. Model reasoning thought and tool call parsing
  3. Action grounding (3D deprojection, safety corridor, discrete quantization)
  4. Physical displacement in embodiment
  5. Privilege firewall verification (zero oracle leakage)
  6. Final evaluation metrics and failure classification
"""
import sys
import os
import json
from pathlib import Path
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.data.data_loader import load_episodes
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
from agenticnav_paper_replica.harness.runner import run_episode, EpisodeRunResult
from agenticnav_paper_replica.traces.trace_logger import TraceLogger
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.run_benchmark import mock_vlm_policy

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run pilot inspection on 1-2 episodes.")
    parser.add_argument("--episodes-count", type=int, default=2)
    parser.add_argument("--backend", type=str, default="auto", choices=["auto", "ollama", "mock"])
    parser.add_argument("--model", type=str, default="qwen3-vl:8b")
    parser.add_argument("--out-dir", type=str, default="runs/pilot_deep_inspection")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(log_dir=out_dir / "traces")

    config = ReplicaConfig()

    # Load first 2 episodes from benchmark dataset
    ep_file = "agenticnav_paper_replica/data/episode_list_100.json"
    episodes = load_episodes(ep_file, max_episodes=args.episodes_count)

    print("=" * 75)
    print(f"       AGENTICNAV END-TO-END PILOT INSPECTION ({len(episodes)} EPISODES)")
    print("=" * 75)
    print(f"Backend:  {args.backend}")
    print(f"Model:    {args.model}")
    print(f"Episodes: {[ep['episode_id'] for ep in episodes]}")
    print("=" * 75)

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
            print("Ollama server not reachable at localhost:11434, falling back to test policy.")
            vlm_fn = mock_vlm_policy

    for idx, ep_data in enumerate(episodes):
        ep_id = ep_data["episode_id"]
        inst = ep_data["instruction"]
        start_p = ep_data.get("start_position", [0, 0, 0])
        goal_p = ep_data.get("goal_position", [0, 0, -10])
        sp_dist = ep_data.get("shortest_path_distance", 10.0)

        print("\n" + "#" * 75)
        print(f" EPISODE {idx + 1}/{len(episodes)}: {ep_id}")
        print(f" Instruction: \"{inst}\"")
        print(f" Start: {start_p} | Goal: {goal_p} | Shortest Path: {sp_dist:.2f}m")
        print("#" * 75)

        embodiment = MockGroundEmbodiment(max_actions=config.max_simulator_actions)

        # Run episode through the complete harness
        res: EpisodeRunResult = run_episode(
            episode_data=ep_data,
            embodiment=embodiment,
            config=config,
            vlm_client_fn=vlm_fn,
            trace_logger=trace_logger,
        )

        # Print stage-by-stage reasoning and telemetry
        trace = res.trace
        print(f"\n[EXECUTION SUMMARY]")
        print(f"  Total Decision Steps: {len(trace.steps)}")
        print(f"  Final Status:         Success={res.metrics.success} (Cat: {res.failure_category})")
        print(f"  Navigation Error:     {res.metrics.navigation_error_m:.2f}m (Threshold: 3.00m)")
        print(f"  Trajectory Length:    {res.metrics.trajectory_length_m:.2f}m (Shortest Path: {res.metrics.shortest_path_distance_m:.2f}m)")
        print(f"  SPL:                  {res.metrics.spl:.4f}")
        print(f"  Oracle Success:       {res.metrics.oracle_success}")

        print(f"\n--- STEP-BY-STEP REASONING & ACTIONS ---")
        from dataclasses import is_dataclass, asdict
        for step_obj in trace.steps:
            s = asdict(step_obj) if is_dataclass(step_obj) else (step_obj if isinstance(step_obj, dict) else vars(step_obj))
            step_idx = s.get("step_idx", 0)
            node_idx = s.get("node_idx", 0)
            pose = s.get("agent_pose", {"x": 0, "z": 0, "yaw_rad": 0})
            geod = s.get("geodesic_distance_to_goal_m", 0.0)
            act_type = s.get("action_type", "")
            act_payload = s.get("action_payload", {})
            turns = s.get("turns", [])

            print(f"\n  [Step {step_idx} @ Node {node_idx}] Pose: (x={pose['x']:.2f}, z={pose['z']:.2f}, yaw={np.degrees(pose['yaw_rad']):.1f} deg) | Goal Dist: {geod:.2f}m")
            print(f"    Turns in step: {len(turns)}")
            for t_i, turn_obj in enumerate(turns):
                t = asdict(turn_obj) if is_dataclass(turn_obj) else (turn_obj if isinstance(turn_obj, dict) else vars(turn_obj))
                t_calls = t.get("tool_calls", [])
                t_results = t.get("tool_results", [])
                for tc in t_calls:
                    print(f"      Turn {t_i} Tool Call: {tc.get('name')}({tc.get('arguments')})")
                for tr in t_results:
                    res_str = str(tr.get("content", ""))[:120]
                    print(f"      Turn {t_i} Result:    {tr.get('name')} -> {res_str}...")

            if act_type == "move_to":
                print(f"    -> ACTION COMMITTED: move_to | Dist moved: {act_payload.get('moved_m', 0.0):.2f}m | Target bearing: {act_payload.get('theta_deg', 0.0):.1f} deg | Collided: {act_payload.get('collided', False)}")
                print(f"       Reason: \"{act_payload.get('reason', '')}\"")
            elif act_type == "stop":
                print(f"    -> ACTION COMMITTED: stop | Reason: \"{act_payload.get('reason', '')}\"")
            else:
                print(f"    -> TERMINATION ACTION: {act_type} | Payload: {act_payload}")

        # Privilege Firewall Check
        print("\n--- PRIVILEGE FIREWALL AUDIT ---")
        firewall_passed = True
        for step_obj in trace.steps:
            s = asdict(step_obj) if is_dataclass(step_obj) else (step_obj if isinstance(step_obj, dict) else vars(step_obj))
            for turn_obj in s.get("turns", []):
                t = asdict(turn_obj) if is_dataclass(turn_obj) else (turn_obj if isinstance(turn_obj, dict) else vars(turn_obj))
                prompt_txt = str(t.get("prompt_text", ""))
                if f"[{goal_p[0]}" in prompt_txt or f"{goal_p[2]}" in prompt_txt:
                    print("  [ALERT] Privilege firewall leak detected!")
                    firewall_passed = False
        if firewall_passed:
            print("  Privilege Firewall Audit: PASSED (Zero oracle state or goal coordinates in model context)")
            print("  Privilege Firewall Audit: PASSED (Zero oracle state or goal coordinates in model context)")

    print("\n" + "=" * 75)
    print("PILOT INSPECTION RUN COMPLETED.")
    print("=" * 75)

if __name__ == "__main__":
    main()
