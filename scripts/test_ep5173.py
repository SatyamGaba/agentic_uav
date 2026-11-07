import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
from agenticnav_paper_replica.harness.runner import run_episode
from agenticnav_paper_replica.traces.trace_logger import TraceLogger
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from scripts.benchmark_r2rce_postfix import run_one

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)

ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "5173")
config = ReplicaConfig()
config.max_simulator_actions = 500
config.max_decision_steps = 25
config.vlm_model = "qwen3.5:4b"

client = OllamaNavClient(model_name=config.vlm_model, temperature=0.0, seed=42)
scene = resolve_scene_path(ep["scene_id"])
emb = HabitatGroundEmbodiment(scene_id=scene, max_actions=500, gpu_device_id=-1)
trace_logger = TraceLogger(Path("audit/dev_20_eval_500/arm_l/traces"))

res = run_one(episode=ep, config=config, vlm_fn=client, trace_logger=trace_logger, model_name=config.vlm_model, embodiment=emb)
with open("audit/dev_20_eval_500/arm_l/episodes/5173.json", "w") as f:
    json.dump(res, f, indent=2, default=str)

print(f"Ep 5173 updated successfully: {res['failure_category']}")
emb.close()
