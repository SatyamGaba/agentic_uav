import os
import sys
import json
import random
import math
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import (
    HabitatGroundEmbodiment,
    resolve_scene_path,
)
from agenticnav_paper_replica.harness.runner import run_episode
from agenticnav_paper_replica.traces.trace_logger import TraceLogger

def run_random_walk_benchmark(episodes_path, output_dir, seed=42, max_actions=25):
    with open(episodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", data)
    
    config = ReplicaConfig()
    config.max_simulator_actions = max_actions
    config.robot_radius = 0.10
    config.base_margin = 0.30
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(out_dir / "traces")
    
    # Sort by scene to minimize reload
    scene_grouped = sorted(episodes, key=lambda e: resolve_scene_path(str(e.get("scene_id", ""))))
    
    rng = random.Random(seed)
    current_scene = None
    emb = None
    results_map = {}
    
    print(f"Running Random-Walk Baseline on {len(episodes)} episodes (seed={seed}, max_actions={max_actions})...")
    
    for idx, ep in enumerate(scene_grouped, start=1):
        ep_id = str(ep["episode_id"])
        scene_id = ep["scene_id"]
        res_scene = resolve_scene_path(str(scene_id))
        scene_base = res_scene.split("/")[-1].split(".")[0]
        
        if res_scene != current_scene:
            if emb is not None:
                emb.close()
            current_scene = res_scene
            emb = HabitatGroundEmbodiment(
                scene_id=res_scene,
                camera_config=None,
                gpu_device_id=-1,
                max_actions=max_actions,
            )
            
        def random_policy(messages, tools):
            # Call stop with 5% probability per turn or if step count > 15
            if rng.random() < 0.08:
                return {
                    "thought": "Random policy: stopping.",
                    "tool_calls": [{"name": "stop", "arguments": {"reason": "Random walk stop"}}]
                }
            # Otherwise random move_to
            view = rng.choice([0, 1, 2, 3])
            u = round(rng.uniform(0.2, 0.8), 2)
            v = round(rng.uniform(0.5, 0.8), 2)
            return {
                "thought": "Random policy: choosing random floor target.",
                "tool_calls": [{
                    "name": "move_to",
                    "arguments": {
                        "view": view,
                        "u": u,
                        "v": v,
                        "reason": f"Random step view {view}"
                    }
                }]
            }
            
        try:
            result = run_episode(
                episode_data=ep,
                embodiment=emb,
                config=config,
                vlm_client_fn=random_policy,
                trace_logger=trace_logger,
            )
            metrics = result.metrics
            res_dict = {
                "episode_id": ep_id,
                "scene_id": scene_id,
                "called_stop": result.called_stop,
                "failure_category": result.failure_category,
                "success": metrics.success,
                "spl": metrics.spl,
                "oracle_success": metrics.oracle_success,
                "navigation_error_m": metrics.navigation_error_m,
                "trajectory_length_m": metrics.trajectory_length_m,
                "actions": metrics.simulator_action_count,
            }
        except Exception as exc:
            res_dict = {
                "episode_id": ep_id,
                "scene_id": scene_id,
                "error": str(exc),
                "failure_category": "infrastructure_failure",
                "success": False,
                "spl": 0.0,
                "oracle_success": False,
                "navigation_error_m": float("nan"),
                "trajectory_length_m": 0.0,
                "actions": 0,
            }
            
        results_map[ep_id] = res_dict
        print(f"[{idx:>3}/100] Ep {ep_id:>4} ({scene_base}): Success={res_dict['success']} | OSR={res_dict['oracle_success']} | NE={res_dict['navigation_error_m']:.2f}m | Actions={res_dict['actions']}")
        
    if emb is not None:
        emb.close()
        
    ordered = [results_map[str(e["episode_id"])] for e in episodes]
    
    sr = np.mean([r["success"] for r in ordered])
    osr = np.mean([r["oracle_success"] for r in ordered])
    spl = np.mean([r["spl"] for r in ordered])
    ne = np.nanmean([r["navigation_error_m"] for r in ordered])
    tl = np.mean([r["trajectory_length_m"] for r in ordered])
    act = np.mean([r["actions"] for r in ordered])
    
    print("\n" + "="*70)
    print("RANDOM-WALK BASELINE RESULTS (ALL 100 EPISODES):")
    print(f"  Episodes:           {len(ordered)}")
    print(f"  Success Rate (SR):  {sr*100:.1f}%")
    print(f"  SPL:                {spl:.4f}")
    print(f"  Oracle SR (OSR):    {osr*100:.1f}%")
    print(f"  Navigation Error:   {ne:.2f} m")
    print(f"  Trajectory Length:  {tl:.2f} m")
    print(f"  Simulator Actions:  {act:.1f}")
    print("="*70)
    
    out_file = out_dir / "random_walk_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "task": "T-A2 Random Walk Baseline",
            "seed": seed,
            "max_actions": max_actions,
            "summary": {
                "success_rate": sr,
                "spl": spl,
                "oracle_success_rate": osr,
                "navigation_error_m": ne,
                "trajectory_length_m": tl,
                "simulator_actions": act,
            },
            "episodes": ordered
        }, f, indent=2)
    print(f"Saved random walk summary to {out_file}")

if __name__ == "__main__":
    ep_file = "agenticnav_paper_replica/data/opennav_r2rce_100.json"
    out_dir = "audit/metric_validation/random_walk"
    run_random_walk_benchmark(ep_file, out_dir, seed=42)

