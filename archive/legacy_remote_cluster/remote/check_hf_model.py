import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
export HF_HOME=/home/jovyan/scratch/hf_cache
python3 -c "
from huggingface_hub import HfApi
api = HfApi()
mid = 'Qwen/Qwen3.5-2B'
try:
    info = api.model_info(mid)
    print('Found model:', info.id, 'sha:', info.sha)
except Exception as e:
    print('Error finding Qwen/Qwen3.5-2B:', e)

mid2 = 'Qwen/Qwen2.5-1.5B-Instruct'
try:
    info2 = api.model_info(mid2)
    print('Found model:', info2.id, 'sha:', info2.sha)
except Exception as e:
    print('Error finding Qwen2.5-1.5B:', e)
"
"""

payload = {"name": "check-hf-model", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

