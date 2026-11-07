import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request
from remote.manage_job import fetch_log

script = """#!/usr/bin/env bash
echo "=== 1. ENVIRONMENT & PATH ==="
echo "PATH: $PATH"
which ollama || echo "ollama not in standard PATH"

echo "=== 2. FINDING OLLAMA BINARY ==="
find /usr /home /opt -name "ollama" -type f 2>/dev/null || true

echo "=== 3. CHECKING RUNNING PROCESSES ==="
ps aux | grep -i ollama | grep -v grep || echo "No ollama process running"

echo "=== 4. CHECKING HTTP PORT 11434 ==="
curl -s http://127.0.0.1:11434/api/version || echo "Ollama port 11434 not listening"

echo "=== 5. CHECKING OLLAMA MODELS DIR ==="
ls -lah /home/jovyan/scratch/ollama_models/manifests/registry.ollama.ai/library/ || true
"""

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={
    "name": "chk-ollama",
    "time_limit_min": 5,
    "script": script
})
print("Job submitted:", res)
jid = res.get("id") or res.get("job_id")
for _ in range(15):
    time.sleep(2)
    log = fetch_log(jid)
    if "FINDING OLLAMA" in log or "ENVIRONMENT" in log:
        print("=== JOB LOG ===\n", log)
        break

