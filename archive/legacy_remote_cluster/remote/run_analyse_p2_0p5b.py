import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
set -e
cd /home/jovyan/scratch/TEST
export HF_HOME=/home/jovyan/scratch/hf_cache

echo "=== STAGE: ANALYSE gpu_0p5b ==="
python3 run_all.py --stage analyse --runs gpu_0p5b --workers 16

echo "=== STAGE: P2 gpu_0p5b ==="
python3 run_all.py --stage p2 --runs gpu_0p5b

echo "=== STATUS SUMMARY ==="
cat status.json
"""

payload = {
    "name": "run-0p5b-analyse-p2",
    "gpu_count": 1,
    "time_limit_min": 60,
    "script": script
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"Submitted Job #{job_id}: Model 1 analyse + p2")

while True:
    time.sleep(10)
    j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
    status = j.get("status")
    print(f"Job #{job_id} {status} (elapsed: {j.get('elapsed_s'):.1f}s)")
    if status in ["succeeded", "failed"]:
        print(f"Final status: {status}, exit_code: {j.get('exit_code')}")
        print("--- LOG ---")
        print(fetch_log(job_id))
        break

