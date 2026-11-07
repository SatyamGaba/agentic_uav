import json

with open("audit/dev_20_eval_500/arm_l/traces/evaluation_traces.jsonl") as f:
    lines = f.readlines()
print("Total lines:", len(lines))
if len(lines) > 1:
    t = json.loads(lines[1])
    print(f"Ep {t['episode_id']} steps: {len(t['steps'])}")
    for si, step in enumerate(t["steps"][:3]):
        print(f"--- Step {si+1} ---")
        for turn in step.get("turns", []):
            role = turn.get("role")
            if role == "assistant":
                tc = turn.get("response", {}).get("tool_calls", [])
                print(f"  Assistant: {tc}")
            elif role == "tool":
                print(f"  Tool ({turn.get('name')}): {turn.get('content')}")
