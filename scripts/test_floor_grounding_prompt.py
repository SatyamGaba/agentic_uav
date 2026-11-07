import urllib.request
import json
import base64
from pathlib import Path
from PIL import Image

# Test prompt with real habitat view_0 image
view0_path = Path(r"C:\Users\Kruti\.gemini\antigravity\brain\845399dd-118d-4b96-a067-6f8e6f44b148\sample_habitat_rgb_view0.png")
if not view0_path.exists():
    # Fallback to obstacle overlay
    view0_path = Path(r"C:\Users\Kruti\.gemini\antigravity\brain\845399dd-118d-4b96-a067-6f8e6f44b148\obstacle_overlay_view0.png")

with open(view0_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# 1. Baseline prompt (current system prompt)
prompt_baseline = """You are a navigation robot.
Available action: move_to(view, u, v, reason).
Select clear floor space or an open doorway to advance forward.
Output JSON:
{"name": "move_to", "arguments": {"view": 0, "u": <float>, "v": <float>, "reason": "<string>"}}"""

# 2. Enhanced prompt with explicit floor coordinate instruction
prompt_enhanced = """You are a navigation robot. Your camera is at eye level (1.25m height).
- The horizon / eye level is at row v = 0.50 (where walls and doors meet).
- The walkable floor is in the lower half of the image (v between 0.65 and 0.85).
Available action: move_to(view, u, v, reason).
To advance forward safely along the path, select an unobstructed point ON THE FLOOR (v in [0.65, 0.85]) in front of the open corridor or doorway.
Output JSON:
{"name": "move_to", "arguments": {"view": 0, "u": <float>, "v": <float>, "reason": "<string>"}}"""

def query_ollama(prompt, think=False):
    payload = {
        "model": "qwen3.5:4b",
        "messages": [
            {"role": "user", "content": prompt, "images": [img_b64]}
        ],
        "stream": False,
        "think": think,
        "options": {"temperature": 0.0, "seed": 42}
    }
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

print("Testing Baseline prompt without thinking...")
try:
    res1 = query_ollama(prompt_baseline, think=False)
    print("Baseline (think=False):", res1.get("message", {}).get("content"))
except Exception as e:
    print("Ollama query failed:", e)

print("\nTesting Enhanced prompt with explicit floor instruction...")
try:
    res2 = query_ollama(prompt_enhanced, think=False)
    print("Enhanced (think=False):", res2.get("message", {}).get("content"))
except Exception as e:
    print("Ollama query failed:", e)

print("\nTesting Enhanced prompt with thinking=True...")
try:
    res3 = query_ollama(prompt_enhanced, think=True)
    print("Enhanced (think=True):", res3.get("message", {}).get("content"))
except Exception as e:
    print("Ollama query failed:", e)
