import json

with open("audit/dev_arm_l_5ep/traces/evaluation_traces.jsonl") as f:
    traces_l = [json.loads(line) for line in f]

print("Keys:", list(traces_l[0].keys()))
for t in traces_l:
    print(f"  Ep {t['episode_id']}: {t.get('failure_category')}, steps={t.get('total_steps')}, actions={t.get('metrics', {}).get('simulator_action_count')}, SR={t.get('metrics', {}).get('success')}, NE={t.get('metrics', {}).get('navigation_error_m', 0.0):.2f}m")
