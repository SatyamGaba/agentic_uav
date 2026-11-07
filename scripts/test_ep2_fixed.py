#!/usr/bin/env python3
"""Run closed-loop test on Episode 2 to verify stop behavior fix."""

import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.harness.runner import run_episode
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient


def main():
    print("=== TESTING CLOSED-LOOP EPISODE 2 WITH FIXED STOP GUIDANCE ===")
    ep_file = ROOT_DIR / "agenticnav_paper_replica/data/episode_list_100.json"
    with open(ep_file, "r") as f:
        all_eps = json.load(f)["episodes"]
    eps = [e for e in all_eps if e.get("scene_id") == "gZ6f7yhEvPG"]
    ep = eps[1]  # Episode 2
    
    print(f"Episode ID: {ep['episode_id']}")
    print(f"Instruction: \"{ep['instruction']}\"")
    
    embodiment = HabitatGroundEmbodiment(scene_id="gZ6f7yhEvPG", gpu_device_id=-1, max_actions=25)
    vlm_client = OllamaNavClient(model_name="qwen3.5:4b", thinking=False, temperature=0.0)
    
    config = ReplicaConfig()
    config.max_decision_steps = 10
    config.max_simulator_actions = 25
    config.success_distance_m = 3.0
    
    res = run_episode(
        episode_data=ep,
        embodiment=embodiment,
        config=config,
        vlm_client_fn=vlm_client,
    )
    
    metrics = res.metrics
    print("\n--- EPISODE OUTCOME ---")
    print(f"Success: {metrics.success}")
    print(f"SPL: {metrics.spl:.4f}")
    print(f"Oracle Success: {metrics.oracle_success}")
    print(f"Navigation Error: {metrics.navigation_error_m:.2f}m")
    print(f"Trajectory Length: {metrics.trajectory_length_m:.2f}m")
    print(f"Failure Category: {res.failure_category}")
    print(f"Total Steps: {res.total_steps}")
    print(f"Simulator Actions: {metrics.simulator_action_count}")
    print(f"Terminated Deadlock: {res.terminated_deadlock}")
    
    embodiment.close()

if __name__ == "__main__":
    main()
