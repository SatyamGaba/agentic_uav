import json

with open("audit/phase9/evaluation_traces.jsonl") as f:
    lines = [json.loads(line) for line in f if line.strip()]

ep2 = lines[1]
print("Episode 2 ID:", ep2["episode_id"])
print("Instruction:", ep2["instruction"])
print("Shortest Path:", ep2["shortest_path_distance_m"])
print("Metrics:", ep2["metrics"])

print("\nStep Traces:")
for s in ep2["steps"]:
    print(f"\n--- Step {s['step_idx']} (dist to goal: {s['geodesic_distance_to_goal_m']:.2f}m) ---")
    for t in s["turns"]:
        r = t.get("response", {})
        thought = r.get("thought", "")
        tools = r.get("tool_calls", [])
        if thought:
            print("  [Thought]:", thought[:150], "...")
        if tools:
            print("  [Tool Calls]:", tools)
        if t.get("role") == "tool":
            print(f"  [Tool Output {t.get('name')}]:", str(t.get("content"))[:100])

