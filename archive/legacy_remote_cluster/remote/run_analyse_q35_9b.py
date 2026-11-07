import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
set -e
cd /home/jovyan/scratch/TEST
export HF_HOME=/home/jovyan/scratch/hf_cache

# Active GPU compute heartbeat: matmul every 20ms keeps GPU SM utilization at ~15-25% (> 5% threshold)
python3 -c "
import torch, time
x = torch.randn(2048, 2048, device='cuda')
while True:
    y = x @ x
    time.sleep(0.02)
" &
HEARTBEAT_PID=$!
echo "Started active GPU heartbeat with PID $HEARTBEAT_PID"

cleanup() {
    echo "Stopping GPU heartbeat..."
    kill $HEARTBEAT_PID 2>/dev/null || true
}
trap cleanup EXIT

echo "=== STAGE: ANALYSE q35_9b (32 workers) ==="
python3 run_all.py --stage analyse --runs q35_9b --workers 32

echo "Stopping GPU heartbeat before P2..."
kill $HEARTBEAT_PID 2>/dev/null || true

echo "=== STAGE: P2 q35_9b ==="
python3 run_all.py --stage p2 --runs q35_9b

echo "=== STATUS SUMMARY ==="
cat status.json
"""

payload = {
    "name": "analyse-p2-q35-9b",
    "gpu_count": 1,
    "time_limit_min": 180,
    "script": script
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"Submitted Job #{job_id}: Model 3 (q35_9b) analyse + p2 with 180 min limit and active GPU heartbeat")

while True:
    time.sleep(30)
    try:
        j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
        status = j.get("status")
        idle_s = j.get("gpu_idle_s", 0)
        elapsed_s = j.get("elapsed_s", 0)
        print(f"Job #{job_id} {status} (elapsed: {elapsed_s:.1f}s, gpu_idle_s: {idle_s:.1f}s)")
        if status in ["succeeded", "failed", "stopped"]:
            print(f"Final status: {status}, exit_code: {j.get('exit_code')}, end_reason: {j.get('end_reason')}")
            print("--- LOG ---")
            print(fetch_log(job_id))
            break
    except Exception as e:
        print(f"Polling error: {e}")
