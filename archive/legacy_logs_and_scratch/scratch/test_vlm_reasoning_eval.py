"""
Evaluates VLM decision-making and reasoning quality with the upgraded AgenticNav context.
Tests whether the model:
1. Deconstructs the multi-stage instruction into milestones.
2. Uses accumulated odometry to avoid premature stops when depth clearance is 2.5m.
3. Continues forward traversal until the destination milestone is reached.
4. Appropriately executes stop() only at the terminal destination.
"""
import sys
import os
import json
import urllib.request
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agenticnav_paper_replica.harness.prompts import SYSTEM_PROMPT, build_step_user_prompt
from agenticnav_paper_replica.harness.tools import get_available_tools
from agenticnav_paper_replica.vlm.ollama_client import parse_tool_calls_from_text

OLLAMA_MODEL = "qwen3.5:latest"

def query_ollama(messages, tools):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 4096,
        }
    }
    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        msg = res.get("message", {})
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls", [])
        if not tool_calls and content:
            tool_calls = parse_tool_calls_from_text(content)
        return content, tool_calls

def test_scenario(scenario_name, instruction, current_node, history_text, accumulated_dist_m, depth_feedback):
    tools = get_available_tools(enable_depth=True, enable_recall=True)
    user_prompt = build_step_user_prompt(
        instruction=instruction,
        current_node=current_node,
        history_text=history_text,
        accumulated_distance_m=accumulated_dist_m,
    )
    if depth_feedback:
        user_prompt += f"\n\n### Current Depth Observation:\n{depth_feedback}"
        
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    print("\n" + "=" * 75)
    print(f" [TEST SCENARIO] {scenario_name}")
    print("=" * 75)
    print(f" Instruction:    \"{instruction}\"")
    print(f" Current Node:   Node {current_node}")
    print(f" Odometry Dist:  {accumulated_dist_m:.2f} meters")
    print(f" Depth Context:  {depth_feedback}")
    print("-" * 75)
    
    t0 = time.time()
    content, tool_calls = query_ollama(messages, tools)
    elapsed = time.time() - t0
    
    print(f" Response Time:  {elapsed:.2f}s")
    print("\n--- MODEL REASONING / THOUGHT ---")
    print(content.strip() if content.strip() else "(No explicit reasoning output)")
    print("\n--- INVOKED TOOL CALLS ---")
    print(json.dumps(tool_calls, indent=2))
    return content, tool_calls

def main():
    instruction = "Exit the room and turn left. Walk forward towards the kitchen and stop near the counter."
    
    # -------------------------------------------------------------
    # Scenario A: Initial Step (Node 1, 0m traversed)
    # Expected: move_to or query_depth to exit room (Milestone 1)
    # -------------------------------------------------------------
    test_scenario(
        scenario_name="Scenario A: Step 1 (Departure from Initial Room)",
        instruction=instruction,
        current_node=1,
        history_text="No previous actions recorded.",
        accumulated_dist_m=0.0,
        depth_feedback="Front View: 2.5m open space toward room exit door.",
    )
    
    # -------------------------------------------------------------
    # Scenario B: THE PREMATURE-STOP TRAP (Node 3, 4.0m traversed)
    # The previous prompt caused 8/9 episodes to stop here because depth was 2.5m!
    # Expected with new prompt:
    # Model must RECOGNIZE it is only 4.0m into route, milestone 3 is in progress,
    # and MUST NOT call stop(). It must call move_to to continue down corridor!
    # -------------------------------------------------------------
    history_node3 = (
        "- Step 1 (Node 1 -> Node 2): [Reason: Exit bedroom door into corridor] [Action: move_to(view=Right, u=0.5, v=0.55) -> Executed (displaced 2.00m, turn -90°)]\n"
        "- Step 2 (Node 2 -> Node 3): [Reason: Turn left into hallway towards kitchen] [Action: move_to(view=Left, u=0.5, v=0.55) -> Executed (displaced 2.00m, turn +90°)]"
    )
    c_b, tc_b = test_scenario(
        scenario_name="Scenario B: Step 3 (The 2.5m Depth Trap - Should NOT Stop!)",
        instruction=instruction,
        current_node=3,
        history_text=history_node3,
        accumulated_dist_m=4.0,
        depth_feedback="Front View: 2.5m open hallway corridor ahead. Left/Right: Walls at 1.6m.",
    )
    
    # -------------------------------------------------------------
    # Scenario C: Arrival at Goal (Node 6, 10.2m traversed, facing counter)
    # Expected: stop(reason=...)
    # -------------------------------------------------------------
    history_node6 = (
        "- Step 1 (Node 1 -> Node 2): [Reason: Exit door] [Action: move_to(view=Right, u=0.5, v=0.55) -> Executed (displaced 2.00m)]\n"
        "- Step 2 (Node 2 -> Node 3): [Reason: Turn left into hallway] [Action: move_to(view=Left, u=0.5, v=0.55) -> Executed (displaced 2.00m)]\n"
        "- Step 3 (Node 3 -> Node 4): [Reason: Walk down hallway] [Action: move_to(view=Front, u=0.5, v=0.55) -> Executed (displaced 2.00m)]\n"
        "- Step 4 (Node 4 -> Node 5): [Reason: Continue down hallway to kitchen entrance] [Action: move_to(view=Front, u=0.5, v=0.55) -> Executed (displaced 2.00m)]\n"
        "- Step 5 (Node 5 -> Node 6): [Reason: Enter kitchen towards counter] [Action: move_to(view=Front, u=0.5, v=0.55) -> Executed (displaced 2.20m)]"
    )
    c_c, tc_c = test_scenario(
        scenario_name="Scenario C: Step 6 (Arrived at Kitchen Counter - Terminal Action)",
        instruction=instruction,
        current_node=6,
        history_text=history_node6,
        accumulated_dist_m=10.2,
        depth_feedback="Front View: Kitchen counter surface directly ahead at 1.2m.",
    )

if __name__ == "__main__":
    main()

