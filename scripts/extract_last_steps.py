import json

with open("audit/phase9/evaluation_traces.jsonl") as f:
    lines = [json.loads(l) for l in f if l.strip()]

def dump_episode_end(ep, max_steps=10):
    print(f"================================================================================")
    print(f"EPISODE: {ep['episode_id']} | Scene: {ep['scene_id']}")
    print(f"Instruction: \"{ep['instruction']}\"")
    print(f"Shortest Path: {ep['shortest_path_distance_m']:.2f}m")
    m = ep.get("metrics", {})
    print(f"Metrics: NE={m.get('navigation_error_m'):.2f}m, Actions={m.get('simulator_actions_taken', m.get('simulator_action_count'))}, Success={m.get('success')}, FailureCat={ep.get('failure_category')}")
    print(f"--------------------------------------------------------------------------------")
    
    # Flatten turns across all steps
    flat_turns = []
    for s_idx, step in enumerate(ep.get("steps", [])):
        dist = step.get("geodesic_distance_to_goal_m", 0.0)
        pose = step.get("agent_pose", {})
        for t in step.get("turns", []):
            flat_turns.append((s_idx + 1, dist, pose, t))
            
    recent = flat_turns[-max_steps:] if len(flat_turns) >= max_steps else flat_turns
    for s_num, dist, pose, t in recent:
        role = t.get("role")
        if role == "assistant":
            resp = t.get("response", {})
            thought = resp.get("thought", "")
            tool_calls = resp.get("tool_calls", [])
            print(f"\n[Step {s_num} | Dist: {dist:.2f}m | Pose: ({pose.get('x'):.2f}, {pose.get('z'):.2f})] ASSISTANT:")
            if thought:
                print(f"  Thought: {thought}")
            for tc in tool_calls:
                print(f"  Tool Call: {tc.get('name')}({json.dumps(tc.get('arguments', {}))})")
        elif role == "tool":
            print(f"[TOOL {t.get('name')}]: {t.get('content')}")
        elif role == "user":
            print(f"[USER PROMPT]: {t.get('content')}")

dump_episode_end(lines[0], max_steps=12)
dump_episode_end(lines[1], max_steps=12)

