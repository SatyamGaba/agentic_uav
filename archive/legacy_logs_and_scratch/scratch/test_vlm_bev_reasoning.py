"""
Tests whether the VLM can effectively perceive and reason about the BEV map image.
Feeds a generated BEV map image + camera views into OllamaNavClient and verifies:
1. Does the model recognize the BEV map (white free space, black walls, cyan path, red position)?
2. Does it accurately identify navigable corridors vs obstacle walls?
3. Does it select safe, forward-progressing actions based on the map geometry?
"""
import sys
import os
import json
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.vlm.ollama_client import OllamaNavClient
from agenticnav_paper_replica.harness.prompts import SYSTEM_PROMPT, build_step_user_prompt
from agenticnav_paper_replica.harness.tools import get_available_tools

def main():
    bev_path = os.path.join(
        os.path.expanduser("~"),
        ".gemini", "antigravity", "brain", "845399dd-118d-4b96-a067-6f8e6f44b148", "bev_map_ep03.png"
    )
    if not os.path.exists(bev_path):
        # Fallback to test_corridor_bev.png or established_bev_test_result.png
        bev_path = os.path.join(
            os.path.expanduser("~"),
            ".gemini", "antigravity", "brain", "845399dd-118d-4b96-a067-6f8e6f44b148", "established_bev_test_result.png"
        )
    
    print(f"Loading BEV Map from: {bev_path}")
    bev_img = np.array(Image.open(bev_path).convert("RGB"))
    print(f"BEV Map shape: {bev_img.shape}, dtype: {bev_img.dtype}")

    # Generate 4 synthetic RGB camera views (indoor corridor colors)
    # View 0 (Front): Open corridor (cream/light gray wall at distance, beige floor)
    front_view = np.full((480, 640, 3), 200, dtype=np.uint8)
    front_view[240:, :] = [180, 160, 140] # floor
    # View 1 (Right): Wall closer by
    right_view = np.full((480, 640, 3), 160, dtype=np.uint8)
    right_view[240:, :] = [180, 160, 140]
    # View 2 (Back): Traversed corridor
    back_view = np.full((480, 640, 3), 190, dtype=np.uint8)
    back_view[240:, :] = [180, 160, 140]
    # View 3 (Left): Wall closer by
    left_view = np.full((480, 640, 3), 160, dtype=np.uint8)
    left_view[240:, :] = [180, 160, 140]

    rgb_views = {0: front_view, 1: right_view, 2: back_view, 3: left_view}

    instruction = "Walk forward through the hallway, pass the doorway on your right, and enter the lounge area."
    history_text = "- Step 1 (Node 1 -> Node 2): [Reason: Move straight along corridor] [Action: move_to(view=0, u=0.5, v=0.55)]\n- Step 2 (Node 2 -> Node 3): [Reason: Continue forward past initial corner] [Action: move_to(view=0, u=0.5, v=0.55)]"

    user_prompt = build_step_user_prompt(
        instruction=instruction,
        current_node=3,
        history_text=history_text,
        accumulated_distance_m=4.20,
    )
    user_prompt += "\n\nAnalyze the attached Bird's-Eye View (BEV) map and camera views:\n1. Where are the walls/obstacles relative to your position (the red node)?\n2. Which direction provides open navigable free space (white area)?\n3. What action will you take to progress towards the lounge?"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": user_prompt,
            "rgb_views": rgb_views,
            "bev_image": bev_img,
        }
    ]

    tools = get_available_tools(enable_depth=True, enable_recall=True)

    client = OllamaNavClient(model_name="qwen3.5:latest")
    print("\nQuerying Ollama with BEV Map image + RGB Views + Context...")
    result = client(messages, tools)

    print("\n" + "=" * 75)
    print(" VLM MULTIMODAL REASONING & BEV UNDERSTANDING RESULT")
    print("=" * 75)
    print("\n--- MODEL REASONING CONTENT ---")
    print(result.get("content", "(No explicit text content)"))
    print("\n--- TOOL CALLS ---")
    print(json.dumps(result.get("tool_calls", []), indent=2))

if __name__ == "__main__":
    main()

