import json

with open("audit/dev_20_eval_500/arm_l/traces/evaluation_traces.jsonl") as f:
    for line in f:
        t = json.loads(line)
        if t["episode_id"] == "8628":
            print("Episode 8628:")
            for s in t["steps"]:
                print(f"Step {s['step_idx']}:")
                for turn in s["turns"]:
                    role = turn.get("role")
                    if role == "assistant":
                        resp = turn.get("response", {})
                        print(f"  Assistant thought: {resp.get('thought')}")
                        print(f"  Assistant tool_calls: {resp.get('tool_calls')}")
                    elif role == "tool":
                        print(f"  Tool {turn.get('name')}: {turn.get('content')}")
                    elif role == "user":
                        print(f"  User msg: {turn.get('content')[:100]}...")

