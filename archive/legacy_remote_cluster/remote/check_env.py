"""
Comprehensive Environment and Hardware Verification Script for Ananta GPU Cluster.
Submits an environment diagnostic job to inspect GPU compute, PyTorch/CUDA,
installed libraries, scratch storage, and datasets.

Usage:
    python remote/check_env.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request
from remote.manage_job import fetch_log

CHECK_SCRIPT = """#!/usr/bin/env bash
set -e
echo "================================================================="
echo "        ANANTA CLUSTER REMOTE ENVIRONMENT DIAGNOSTIC REPORT      "
echo "================================================================="

echo -e "\\n[1. GPU ACCELERATOR & DRIVER INFO]"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu --format=csv

echo -e "\\n[2. PYTHON & PYTORCH CUDA ACCELERATION]"
python3 -c "
import sys
print(f'Python Executable: {sys.executable}')
print(f'Python Version   : {sys.version.split()[0]}')
try:
    import torch
    print(f'PyTorch Version  : {torch.__version__}')
    print(f'CUDA Available   : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'Device Name      : {torch.cuda.get_device_name(0)}')
        print(f'Device Count     : {torch.cuda.device_count()}')
        print(f'Compute Cap      : {torch.cuda.get_device_capability(0)}')
        x = torch.randn(1000, 1000, device='cuda')
        y = x @ x
        print(f'GPU Matrix Mult  : OK (Allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB)')
except Exception as e:
    print(f'PyTorch/CUDA Check Failed: {e}')
"

echo -e "\\n[3. CRITICAL PACKAGES CHECK]"
python3 -c "
pkgs = ['numpy', 'scipy', 'PIL', 'google.genai', 'habitat', 'websockets', 'requests']
for p in pkgs:
    try:
        mod = __import__(p)
        ver = getattr(mod, '__version__', 'Installed (no __version__)')
        print(f'  {p:<15}: {ver}')
    except ImportError:
        print(f'  {p:<15}: NOT INSTALLED')
"

echo -e "\\n[4. DISK & SCRATCH MOUNT CHECK]"
df -h /home/jovyan/scratch

echo -e "\\n[5. DATASETS & CACHES UNDER SCRATCH]"
ls -la /home/jovyan/scratch/ | head -n 25

echo -e "\\n[6. SCANNING COMMON MATTERPORT3D / HABITAT DATASET PATHS]"
python3 -c "
import os
candidates = [
    '/home/jovyan/scratch/scene_datasets',
    '/home/jovyan/scratch/data',
    '/home/jovyan/scratch/mp3d',
    '/home/jovyan/data',
    '/data/scene_datasets',
    '/data/mp3d',
    '/datasets/mp3d',
    '/datasets',
    '/workspace/data'
]
found = False
for d in candidates:
    if os.path.exists(d):
        try:
            items = os.listdir(d)[:8]
            print(f'  [FOUND] {d}: {items}')
            found = True
        except Exception as e:
            print(f'  [FOUND] {d} (Permission error: {e})')
if not found:
    print('  [INFO] No Matterport3D datasets found in standard directories.')
"

echo -e "\\n[7. ACTIVE PROCESSES & OLLAMA CHECK]"
ps aux | grep -v grep | grep -E 'ollama|python|vllm' || echo 'No background inference servers running.'

echo -e "\\n================================================================="
echo "                   DIAGNOSTIC COMPLETED                          "
echo "================================================================="
"""

def main():
    print("=" * 70)
    print("      CHECKING REMOTE ENVIRONMENT ON ANANTA GPU CLUSTER           ")
    print("=" * 70)
    
    payload = {
        "name": "env-diagnostic-check",
        "gpu_count": 1,
        "time_limit_min": 5,
        "script": CHECK_SCRIPT
    }

    try:
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
    except Exception as e:
        print(f"[ERROR] Could not submit diagnostic job: {e}")
        print("\nNote: If you receive HTTP 403 Forbidden:")
        print("  1. Please log in to the campus network portal at: https://campnet.bits-goa.ac.in:8090/")
        print("  2. If connecting from outside campus, ensure FortiClient VPN is active.")
        sys.exit(1)

    job_id = res.get("job_id") or res.get("id")
    print(f"[SUCCESS] Diagnostic job submitted! (Job ID: #{job_id})")
    print("Waiting for job execution and gathering report...\n")

    for _ in range(30):
        time.sleep(3)
        try:
            j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
            status = j.get("status")
            if status in ["succeeded", "failed"]:
                log = fetch_log(job_id)
                print(log)
                if status == "failed":
                    print(f"\n[WARNING] Diagnostic job finished with status: {status} (Exit code: {j.get('exit_code')})")
                else:
                    print("\n[PASS] Remote environment check completed successfully!")
                return
        except Exception as e:
            print(f"Polling: {e}")

    print(f"Job is still running. View output anytime with: python remote/manage_job.py --id {job_id}")

if __name__ == "__main__":
    main()

