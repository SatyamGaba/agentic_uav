import sys, os, time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.harness.tools import get_available_tools
from agenticnav_paper_replica.harness.prompts import SYSTEM_PROMPT, build_step_user_prompt

print("Initializing OllamaNavClient for qwen3.5:latest...")
client = OllamaNavClient(model_name="qwen3.5:latest")
print(f"Model configured: {client.is_configured()} | Supports vision: {client.supports_vision}")

tools = get_available_tools(enable_depth=True, enable_recall=True)
dummy_img = Image.new("RGB", (64, 64), color="blue")
rgb_views = {0: dummy_img, 1: dummy_img, 2: dummy_img, 3: dummy_img}
bev_img = Image.new("RGB", (64, 64), color="gray")

prompt = build_step_user_prompt(
    instruction="Walk past the bed and out the door. Take a right down the hallway and stop near the table.",
    current_node=1,
    history_text="None. You are at the starting position.",
    accumulated_distance_m=0.0
)

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": prompt,
        "rgb_views": rgb_views,
        "bev_image": bev_img,
    }
]

print("Sending request to local Qwen 3.5...")
t0 = time.time()
res = client(messages, tools)
elapsed = time.time() - t0
print(f"Done in {elapsed:.2f}s!")
print("THOUGHT:\n", res.get("thought", "").strip())
print("TOOL CALLS:\n", res.get("tool_calls", []))

