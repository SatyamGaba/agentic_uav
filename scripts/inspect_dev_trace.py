import json

with open("audit/dev_20_eval/traces/evaluation_traces.jsonl") as f:
    t = json.loads(f.readline())

print(f"Episode ID: {t.get('episode_id')}")
steps = t.get("steps", [])
print(f"Total steps: {len(steps)}")

for idx, s in enumerate(steps):
    print(f"\n--- STEP {idx+1} ---")
    for k, v in s.items():
        if k == "turn_log":
            print(f"  turn_log ({len(v)} turns):")
            for t_idx, turn in enumerate(v):
                print(f"    Turn {t_idx+1}: {turn}")
        elif k not in ("rgb_views", "depth_views", "bev_image"):
            print(f"  {k}: {v}")
