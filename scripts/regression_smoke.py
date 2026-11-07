"""
Phase 10 Verification Script: Regression Smoke Test (scripts/regression_smoke.py)

Runs a 3-episode mini-benchmark (episodes 1, 2, 3) on the locked config
and asserts that the pipeline completes without regressions.
"""

import sys
import os
import json
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.harness.runner import run_episode, EpisodeRunResult
from agenticnav_paper_replica.traces.trace_logger import TraceLogger


def run_regression_smoke():
    print("=" * 80)
    print("PHASE 10: REGRESSION SMOKE TEST (EPISODES 1, 2, 3)")
    print("=" * 80)

    ep_file = PROJECT_ROOT / "agenticnav_paper_replica/data/episode_list_100.json"
    with open(ep_file, "r") as f:
        episodes = json.load(f)["episodes"][:3]

    scene_id = "gZ6f7yhEvPG"
    config = ReplicaConfig()
    config.max_simulator_actions = 15

    print(f"Initializing persistent Habitat-Sim on {scene_id}...")
    embodiment = HabitatGroundEmbodiment(scene_id=scene_id, gpu_device_id=-1, max_actions=config.max_simulator_actions)

    print("Initializing OllamaNavClient (qwen3.5:4b)...")
    vlm_client = OllamaNavClient(model_name="qwen3.5:4b", thinking=False, temperature=0.0)

    smoke_dir = PROJECT_ROOT / "audit" / "phase10" / "smoke_traces"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(smoke_dir)

    all_passed = True
    results = []

    for idx, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        print(f"\n[Smoke {idx+1}/3] Running {ep_id}...")
        ep_copy = dict(ep)
        ep_copy["scene_id"] = scene_id

        t0 = time.time()
        res: EpisodeRunResult = run_episode(
            episode_data=ep_copy,
            embodiment=embodiment,
            config=config,
            vlm_client_fn=vlm_client,
            trace_logger=trace_logger,
        )
        dur = time.time() - t0

        pass_ep = (res.total_steps >= 1 and res.metrics.navigation_error_m >= 0.0)
        all_passed = all_passed and pass_ep

        print(f"  Finished in {dur:.2f}s | Steps: {res.total_steps} | Actions: {res.metrics.simulator_action_count} | NE: {res.metrics.navigation_error_m:.2f}m")
        results.append({
            "episode_id": ep_id,
            "success": bool(res.metrics.success),
            "spl": float(res.metrics.spl),
            "ne_m": float(res.metrics.navigation_error_m),
            "steps": int(res.total_steps),
            "actions": int(res.metrics.simulator_action_count),
            "duration_s": round(dur, 2),
            "pass": pass_ep,
        })

    embodiment.close()

    smoke_summary = {
        "status": "PASS" if all_passed else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "results": results,
    }

    out_file = PROJECT_ROOT / "audit" / "phase10" / "smoke_summary.json"
    with open(out_file, "w") as f:
        json.dump(smoke_summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"REGRESSION SMOKE RESULT: {smoke_summary['status']}")
    print(f"Summary written to: {out_file}")
    print("=" * 80)

    return all_passed


if __name__ == "__main__":
    ok = run_regression_smoke()
    sys.exit(0 if ok else 1)

