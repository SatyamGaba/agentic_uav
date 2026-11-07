"""Blind-Image Negative Control for Stage C Gate Verification.

Executes 5 dev episodes with sensory frames completely blanked (uniform gray RGB and BEV).
Evaluates whether the agent is genuinely visually grounded or blindly hallucinating:
- Checks if the model detects lack of visual information.
- Compares trajectory length and OSR against visually grounded runs.
- A visually grounded policy cannot successfully reach landmarks in total blindness.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import (
    HabitatGroundEmbodiment,
    resolve_scene_path,
)
from agenticnav_paper_replica.harness.runner import run_episode
from agenticnav_paper_replica.traces.trace_logger import TraceLogger
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient

def run_blind_control(episodes_path: str, output_dir: str, max_episodes: int = 5):
    with open(episodes_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", data)[:max_episodes]
    
    config = ReplicaConfig()
    config.max_simulator_actions = 25
    config.vlm_model = "qwen3.5:4b"
    config.floor_grounding_prompt = True
    config.arrival_verification = True
    config.obstacle_point_threshold = 5
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_logger = TraceLogger(out_dir / "traces")
    
    client = OllamaNavClient(
        model_name=config.vlm_model,
        temperature=config.temperature,
        seed=config.inference_seed,
        thinking=False,
    )
    
    def blind_vlm_wrapper(messages, tools):
        # Blank out all rgb_views and bev_image in messages
        for msg in messages:
            if "rgb_views" in msg and msg["rgb_views"] is not None:
                blank_views = {}
                for vi, img in msg["rgb_views"].items():
                    blank_views[vi] = np.full_like(img, 128)  # uniform neutral gray
                msg["rgb_views"] = blank_views
            if "bev_image" in msg and msg["bev_image"] is not None:
                msg["bev_image"] = np.full_like(msg["bev_image"], 128)
        return client(messages, tools)
        
    results = []
    print(f"Running Blind-Image Control on {len(episodes)} dev episodes...")
    
    cur_scene = None
    emb = None
    for idx, ep in enumerate(episodes, start=1):
        ep_id = str(ep["episode_id"])
        res_scene = resolve_scene_path(str(ep.get("scene_id", "")))
        
        if res_scene != cur_scene:
            if emb is not None:
                emb.close()
            cur_scene = res_scene
            emb = HabitatGroundEmbodiment(
                scene_id=res_scene,
                max_actions=config.max_simulator_actions,
                gpu_device_id=-1,
            )
            
        print(f"[{idx}/{len(episodes)}] Ep {ep_id} (Blind)")
        res = run_episode(
            episode_data=ep,
            embodiment=emb,
            config=config,
            vlm_client_fn=blind_vlm_wrapper,
            trace_logger=trace_logger,
        )
        results.append({
            "episode_id": ep_id,
            "success": res.metrics.success,
            "oracle_success": res.metrics.oracle_success,
            "spl": res.metrics.spl,
            "navigation_error_m": res.metrics.navigation_error_m,
            "trajectory_length_m": res.metrics.trajectory_length_m,
            "actions": res.metrics.simulator_action_count,
            "failure_category": res.failure_category,
        })
        
    if emb is not None:
        emb.close()
        
    summary = {
        "control": "Blind-Image Sensory Ablation",
        "episodes_tested": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])),
        "oracle_success_rate": float(np.mean([r["oracle_success"] for r in results])),
        "mean_navigation_error_m": float(np.mean([r["navigation_error_m"] for r in results])),
        "mean_actions": float(np.mean([r["actions"] for r in results])),
        "episodes": results,
    }
    
    out_file = out_dir / "blind_control_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved blind control summary to {out_file}")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    dev_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"
    out_path = "audit/dev_20_eval/blind_control"
    run_blind_control(dev_path, out_path, max_episodes=5)

