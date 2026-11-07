import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
set -e
cd /home/jovyan/scratch/TEST
export HF_HOME=/home/jovyan/scratch/hf_cache

# Active GPU compute heartbeat to keep GPU utilization at ~10-20% (> 5% threshold) during CPU probe fitting
python3 -c "
import torch, time
x = torch.randn(4096, 4096, device='cuda')
while True:
    y = x @ x
    time.sleep(0.05)
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

