import json

with open("audit/dev_20_eval/traces/evaluation_traces.jsonl") as f:
    for line in f:
        t = json.loads(line)
        if str(t.get("episode_id")) == "8628":
            print("Found ep 8628:")
            for s in t.get("steps", []):
                print(f"Step {s.get('step_idx')}, Node {s.get('node_idx')}:")
                for turn in s.get("turns", []):
                    role = turn.get("role")
                    resp = turn.get("response", {})
                    tc = resp.get("tool_calls") if isinstance(resp, dict) else None
                    content = turn.get("content")
                    print(f"  [{role}] tc={tc} | content={str(content)[:120]}")

