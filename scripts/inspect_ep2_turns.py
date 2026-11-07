#!/usr/bin/env python3
"""Inspect turns of Episode 2."""
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.harness.runner import run_episode
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.traces.trace_logger import TraceLogger

def main():
    ep_file = ROOT_DIR / "agenticnav_paper_replica/data/episode_list_100.json"
    with open(ep_file, "r") as f:
        all_eps = json.load(f)["episodes"]
    eps = [e for e in all_eps if e.get("scene_id") == "gZ6f7yhEvPG"]
    ep = eps[1]
    
    embodiment = HabitatGroundEmbodiment(scene_id="gZ6f7yhEvPG", gpu_device_id=-1, max_actions=25)
    vlm_client = OllamaNavClient(model_name="qwen3.5:4b", thinking=False, temperature=0.0)
    config = ReplicaConfig()
    config.max_decision_steps = 10
    config.max_simulator_actions = 25
    
    logger = TraceLogger(ROOT_DIR / "audit" / "phase9" / "ep2_inspect")
    res = run_episode(ep, embodiment, config, vlm_client, trace_logger=logger)
    
    trace = res.trace
    print(f"\nTrace recorded {len(trace.steps)} steps:")
    for s in trace.steps:
        print(f"\n--- STEP {s.step_idx} (Dist to goal: {s.geodesic_distance_to_goal_m:.2f}m) ---")
        print(f"Action Type: {s.action_type}")
        print(f"Action Payload: {s.action_payload}")
        print("Turns:")
        for t in s.turns:
            print(f"  {t.get('role')}: {t.get('content') or t.get('response')}")
            
    embodiment.close()

if __name__ == "__main__":
    main()

