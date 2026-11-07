import sys
from config import DASHBOARD_API, api_request

cmd = """#!/bin/bash
set -e
echo "=== CHECKING OLLAMA MODELS ==="
ls -lah /home/jovyan/scratch/ollama_models || true
echo "=== CHECKING SYSTEM HF OR VLLM MODELS ==="
find /tmp /home/jovyan -name "*Qwen*" 2>/dev/null || echo "No Qwen models found"
"""

payload = {
    "name": "check-existing-models",
    "gpu_count": 1,
    "time_limit_min": 5,
    "script": cmd
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
print("Job Submitted:", res)

