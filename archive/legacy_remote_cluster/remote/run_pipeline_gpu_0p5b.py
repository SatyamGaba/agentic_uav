import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
set -e
cd /home/jovyan/scratch/TEST
export HF_HOME=/home/jovyan/scratch/hf_cache

echo "=== STAGE: DOWNLOAD gpu_0p5b ==="
python3 run_all.py --stage download --runs gpu_0p5b

echo "=== STAGE: LOADTEST gpu_0p5b ==="
python3 run_all.py --stage loadtest --runs gpu_0p5b

echo "=== STAGE: SMOKE gpu_0p5b ==="
python3 run_all.py --stage smoke --runs gpu_0p5b

echo "=== STAGE: PILOT gpu_0p5b ==="
python3 run_all.py --stage pilot --runs gpu_0p5b

echo "=== STAGE: PREFLIGHT ==="
python3 run_all.py --stage preflight

echo "=== STAGE: TREE gpu_0p5b ==="
python3 run_all.py --stage tree --runs gpu_0p5b

echo "=== STATUS SUMMARY ==="
cat status.json
"""

payload = {
    "name": "run-0p5b-tree",
    "gpu_count": 1,
    "time_limit_min": 60,
    "script": script
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"Submitted Job #{job_id}: Model 1 (gpu_0p5b) pipeline through tree")

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

