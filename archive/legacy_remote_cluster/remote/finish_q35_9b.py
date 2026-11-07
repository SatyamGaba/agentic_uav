import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
set -e
cd /home/jovyan/scratch/TEST
export HF_HOME=/home/jovyan/scratch/hf_cache

# Start GPU heartbeat to keep GPU active during CPU probe fitting
python3 -c "
import torch, time
x = torch.ones((100, 100), device='cuda')
while True:
    time.sleep(10)
    x = x + 1.0
" &
HEARTBEAT_PID=$!
echo "Started GPU heartbeat with PID $HEARTBEAT_PID"

cleanup() {
    echo "Stopping GPU heartbeat..."
    kill $HEARTBEAT_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "=== STAGE: ANALYSE q35_9b (32 workers) ==="
python3 run_all.py --stage analyse --runs q35_9b --workers 32

echo "=== STAGE: P2 q35_9b ==="
python3 run_all.py --stage p2 --runs q35_9b

echo "=== STATUS SUMMARY ==="
cat status.json
"""

payload = {
    "name": "finish-q35-9b",
    "gpu_count": 1,
    "time_limit_min": 60,
    "script": script
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"Submitted Job #{job_id}: Finish Model 3 (q35_9b) analyse + p2")

while True:
    time.sleep(15)
    j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
    status = j.get("status")
    print(f"Job #{job_id} {status} (elapsed: {j.get('elapsed_s'):.1f}s)")
    if status in ["succeeded", "failed", "stopped"]:
        print(f"Final status: {status}, exit_code: {j.get('exit_code')}")
        print("--- LOG ---")
        print(fetch_log(job_id))
        break

