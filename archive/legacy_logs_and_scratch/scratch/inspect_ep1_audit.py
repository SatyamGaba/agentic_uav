import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log

script_body = """#!/usr/bin/env bash
python3 -c "
import json
with open('/home/jovyan/scratch/agenticnav_paper_replica/runs/eval_10ep/r2r_ce_val_unseen_0001/audit_eval_r2r_ce_val_unseen_0001.json') as f:
    d = json.load(f)
for s in d.get('step_evaluations', []):
    print('ACTION:', s.get('action_type'))
    print('RAW THOUGHT:', repr(s.get('raw_thought')))
"
"""

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={
    "name": "chk-ep1",
    "gpu_count": 1,
    "time_limit_min": 5,
    "script": script_body
})
print("SUBMITTED:", res)
job_id = res.get("id") or res.get("job_id")
for _ in range(15):
    time.sleep(2)
    log = fetch_log(job_id)
    if "ACTION:" in log or "error" in log.lower() or "traceback" in log.lower():
        print("LOG RESULT:\n", log)
        break

