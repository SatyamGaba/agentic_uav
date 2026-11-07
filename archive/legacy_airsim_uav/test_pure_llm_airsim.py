import sys
import os
import time

sys.path.insert(0, '/home/jovyan/scratch')
from framework import create_adapter, create_backend, create_environment

print("1. Connecting to AirSim & Ollama (qwen3-vl:8b)...")
backend = create_backend("ollama", model_name="qwen3-vl:8b", base_url="http://127.0.0.1:11434")
adapter = create_adapter("direct_pixel", backend=backend)
env = create_environment("airsim")

from framework.types import EpisodeGoal
goal = EpisodeGoal(
    episode_id=1,
    instruction="Proceed forward towards the tall modern building with windows.",
    goal_pos=[30.0, 0.0, 5.0],
    gt_trajectory=[[0.0, 0.0, 5.0], [30.0, 0.0, 5.0]],
    max_steps=5
)

print("\n2. Resetting AirSim to start pose...")
obs = env.reset(goal)
print(f"Captured live frame: shape={obs.rgb_shape}, depth={obs.depth_shape}, b64_len={len(obs.rgb_b64) if obs.rgb_b64 else 0}")

print("\n3. Sending live frame + instruction to Qwen3-VL...")
t0 = time.time()
# Let's inspect backend response directly
messages = [
    {"role": "system", "content": "You are a drone navigator. Look at the camera view and output JSON: {\"u\": float, \"v\": float, \"is_stop\": bool, \"reason\": str}."},
    {"role": "user", "content": f"Instruction: {goal.instruction}", "images": [obs.rgb_b64]}
]
raw_resp = backend.chat(messages, max_tokens=1536)
print(f"Latency: {raw_resp.latency_s}s")
print(f"Raw Response Dict: {raw_resp.raw_response}")
print(f"Content: '{raw_resp.content}'")
print(f"Thinking: '{raw_resp.thinking}'")

action = adapter.step(obs)
print(f"\nAdapter Action: u={action.u:.3f}, v={action.v:.3f}, stop={action.is_stop}, reason='{action.reasoning}'")
print(f"Action 3D: dx={action.dx:.2f}, dy={action.dy:.2f}, dz={action.dz:.2f}, yaw={action.yaw_delta:.3f}")

print("\n4. Executing flight in AirSim...")
next_obs, done, info = env.step(action)
print(f"Flight complete. New position: {next_obs.pose.to_dict()}")
env.close()
