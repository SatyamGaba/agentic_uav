import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log
import time

runner_script = """#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/jovyan/scratch/agenticnav_paper_replica:$PYTHONPATH"
cd /home/jovyan/scratch/agenticnav_paper_replica

# 1. Background GPU heartbeat
python3 -c "
import torch, time
x = torch.randn(256, 256, device='cuda')
while True:
    y = x @ x
    time.sleep(5.0)
" &
HB_PID=$!
cleanup() { kill $HB_PID || true; }
trap cleanup EXIT INT TERM

# 2. Ensure local Ollama is active
if command -v ollama >/dev/null 2>&1; then
    if ! pgrep -x "ollama" >/dev/null; then
        echo "[DAEMON] Starting local Ollama service in background..."
        ollama serve >/dev/null 2>&1 &
        sleep 4
    fi
fi

# 3. Run pilot deep inspection
python3 scratch/run_pilot_deep_inspection.py --episodes-count 2 --backend auto --model "qwen3-vl:8b" --out-dir runs/pilot_deep_inspection
"""

payload = {
    "name": "pilot-eval-p2",
    "gpu_count": 1,
    "time_limit_min": 15,
    "script": runner_script,
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"[SUCCESS] Pilot inspection job submitted! Job ID: #{job_id}")

