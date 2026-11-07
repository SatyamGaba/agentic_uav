import json

with open('/mnt/c/Users/Kruti/Desktop/PROG/UAV/audit/phase9/evaluation_traces.jsonl') as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Completed episodes so far: {len(lines)}")
for i, ep in enumerate(lines):
    m = ep.get('metrics', {})
    steps = len(ep.get('steps', []))
    print(f"  [{i+1:02d}] {ep['episode_id']}: steps={steps}, actions={m.get('simulator_actions_taken', m.get('simulator_action_count'))}, NE={m.get('navigation_error_m', 0):.2f}m, success={m.get('success')}")

