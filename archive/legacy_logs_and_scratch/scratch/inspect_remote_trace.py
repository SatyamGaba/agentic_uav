import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log
import time

script = """#!/usr/bin/env bash
python3 - << 'PYEOF'
import json

trace_file = "/home/jovyan/scratch/agenticnav_paper_replica/runs/eval-opt-b3/traces/evaluation_traces.jsonl"
try:
    with open(trace_file) as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            ep_id = data.get("episode_id")
            inst = data.get("instruction")
            steps = data.get("steps", [])
            print("=" * 60)
            print("EPISODE", i + 1, "ID:", ep_id)
            print("Instruction:", inst)
            print("Steps count:", len(steps))
            for s in steps:
                print("  Step:", s.get("step_idx"), "Action:", s.get("action_type"), "Payload:", s.get("action_payload"))
                for t_idx, t in enumerate(s.get("turns", [])):
                    print("    Turn", t_idx, "tool_calls:", t.get("tool_calls"))
                    print("    Turn", t_idx, "tool_results:", t.get("tool_results"))
            if i >= 1:
                break
except Exception as e:
    print("Error reading trace:", e)
PYEOF
"""

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={"name": "inspect-trace", "time_limit_min": 5, "script": script})
jid = res.get("id")
print("Submitted inspect job:", jid)
for _ in range(15):
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break
