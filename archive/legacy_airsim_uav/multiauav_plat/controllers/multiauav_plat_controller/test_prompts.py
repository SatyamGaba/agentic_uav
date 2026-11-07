"""
MultiUAV-Plat / Agent4Drone VLM Prompt Benchmark Runner

Usage:
    python test_prompts.py
"""

import sys
import json
from perception import VLMClient

BENCHMARK_PROMPTS = [
    "move towards the manor and focus on its top left window",
    "find the red tesla model 3 parked near the building and inspect its front right wheel",
    "search for the windmill tower, approach it and orbit 360 degrees",
    "locate the cardboard box on the ground and descend to inspect it"
]

def run_benchmark():
    print("=" * 70)
    print("       MultiUAV-Plat / Agent4Drone VLM Benchmark Prompt Suite       ")
    print("=" * 70)

    client = VLMClient()
    provider = "lmstudio"
    api_key = "http://127.0.0.1:1234/v1/chat/completions"
    model_name = "qwen/qwen3-vl-4b"

    for idx, prompt in enumerate(BENCHMARK_PROMPTS, 1):
        print(f"\n[Test #{idx}] Prompt: \"{prompt}\"")
        subgoals = client.plan_mission(prompt, provider, api_key, model_name)
        print(f" -> Decomposed Sub-Goals ({len(subgoals)}):")
        print(json.dumps(subgoals, indent=2))

if __name__ == "__main__":
    run_benchmark()
