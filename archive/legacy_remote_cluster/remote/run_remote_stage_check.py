from config import DASHBOARD_API, api_request
from manage_job import fetch_log
import time

script = """#!/bin/bash
set -e
cd /home/jovyan/scratch/TEST
export HF_HOME=/home/jovyan/scratch/hf_cache
mkdir -p "$HF_HOME"
python3 run_all.py --stage check
"""

payload = {
    "name": "run-stage-check",
    "gpu_count": 1,
    "time_limit_min": 5,
    "script": script
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"Submitted Stage Check: Job #{job_id}")

for _ in range(30):
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
    if j.get("status") in ["succeeded", "failed"]:
        print(f"Job #{job_id} {j.get('status')} (Exit: {j.get('exit_code')})")
        print("--- LOG ---")
        print(fetch_log(job_id))
        break

