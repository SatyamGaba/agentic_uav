import sys
import os
from config import DASHBOARD_API, api_request

cmd = """#!/bin/bash
set -e
echo "=== GPU INFORMATION ==="
nvidia-smi
echo "=== PYTHON & PACKAGES ==="
python3 -V
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA available:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
python3 -c "import transformers; print('Transformers:', transformers.__version__)" 2>/dev/null || echo "Transformers not installed in base python"
echo "=== SCRATCH DIRECTORY ==="
mkdir -p /home/jovyan/scratch/TEST
ls -lah /home/jovyan/scratch
echo "=== CHECKING EXISTING HUGGINGFACE MODELS ==="
find /home/jovyan/.cache/huggingface -maxdepth 3 2>/dev/null || echo "No ~/.cache/huggingface"
find /home/jovyan/scratch -maxdepth 4 -name "*models--*" 2>/dev/null || echo "No models in scratch"
echo "=== DISK USAGE ==="
df -h /home/jovyan /home/jovyan/scratch
"""

payload = {
    "name": "inspect-remote-env",
    "gpu_count": 1,
    "time_limit_min": 5,
    "script": cmd
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
print("Job Submitted:", res)

