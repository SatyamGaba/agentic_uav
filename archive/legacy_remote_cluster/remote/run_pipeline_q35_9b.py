import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
set -e
cd /home/jovyan/scratch/TEST
export HF_HOME=/home/jovyan/scratch/hf_cache

echo "=== STAGE: DOWNLOAD q35_9b ==="
python3 run_all.py --stage download --runs q35_9b

echo "=== STAGE: LOADTEST q35_9b ==="
python3 run_all.py --stage loadtest --runs q35_9b

echo "=== STAGE: PILOT q35_9b ==="
python3 run_all.py --stage pilot --runs q35_9b

echo "=== STAGE: PREFLIGHT ==="
python3 run_all.py --stage preflight --force

echo "=== STAGE: TREE q35_9b ==="
python3 run_all.py --stage tree --runs q35_9b

echo "=== STAGE: ANALYSE q35_9b ==="
python3 run_all.py --stage analyse --runs q35_9b --workers 16

echo "=== STAGE: P2 q35_9b ==="
python3 run_all.py --stage p2 --runs q35_9b

echo "=== STATUS SUMMARY ==="
cat status.json
"""

payload = {
    "name": "run-q35-9b-full",
    "gpu_count": 1,
    "time_limit_min": 120,
    "script": script
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"Submitted Job #{job_id}: Model 3 (q35_9b / Qwen3.5-9B) full pipeline")

while True:
    time.sleep(15)
    j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
    status = j.get("status")
    print(f"Job #{job_id} {status} (elapsed: {j.get('elapsed_s'):.1f}s)")
    if status in ["succeeded", "failed"]:
        print(f"Final status: {status}, exit_code: {j.get('exit_code')}")
        print("--- LOG ---")
        print(fetch_log(job_id))
        break

