import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
cd /home/jovyan/scratch/TEST/tool_alloc
if [ -d results/pilot_gpu_0p5b ]; then
    ls -la results/pilot_gpu_0p5b
    cat results/pilot_gpu_0p5b/parser_report.txt || true
    cat results/pilot_gpu_0p5b/audit.log || true
    python3 -c "
import json
p = json.load(open('results/pilot_gpu_0p5b/pilot.json'))
print('Keys:', list(p.keys()))
for k, v in p.items():
    if k not in ['instances']:
        print(f'{k}: {v}')
"
fi
"""

payload = {"name": "inspect-pilot", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

