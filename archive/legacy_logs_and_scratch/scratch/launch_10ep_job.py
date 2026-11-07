import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log

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

# 2. Ensure local Ollama is active with correct models directory
export OLLAMA_MODELS="/home/jovyan/scratch/ollama_models"
if command -v ollama >/dev/null 2>&1; then
    if ! pgrep -x "ollama" >/dev/null; then
        echo "[DAEMON] Starting local Ollama service in background..."
        ollama serve >/dev/null 2>&1 &
    fi
fi

# 3. Wait up to 30s for Ollama server to report ready
echo "[DAEMON] Waiting for Ollama HTTP API (port 11434)..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
        echo "[DAEMON] Ollama server is live and ready!"
        break
    fi
    sleep 1
done

# Verify model presence
curl -s http://127.0.0.1:11434/api/tags || true

# 4. Run 10-episode benchmark with full step-by-step diagnostic tracking
python3 scratch/run_pilot_step_by_step_eval.py \\
    --episodes-count 10 \\
    --mode verbose \\
    --backend ollama \\
    --model "qwen3-vl:8b" \\
    --out-dir runs/eval_10ep
"""

payload = {
    "name": "exp-s10-m",
    "gpu_count": 1,
    "time_limit_min": 60,
    "script": runner_script,
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"[SUCCESS] 10-episode benchmark job submitted! Job ID: #{job_id}")

