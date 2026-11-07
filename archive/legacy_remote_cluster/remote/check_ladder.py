import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
export HF_HOME=/home/jovyan/scratch/hf_cache
python3 << 'EOF'
from huggingface_hub import HfApi
api = HfApi()
for mid in ['Qwen/Qwen3.5-9B', 'Qwen/Qwen3.5-4B', 'Qwen/Qwen3.5-0.8B']:
    try:
        info = api.model_info(mid)
        print('Found:', info.id, 'sha:', info.sha)
    except Exception as e:
        print('Failed', mid, e)
EOF
"""
payload = {'name': 'check-ladder-models', 'gpu_count': 1, 'time_limit_min': 5, 'script': script}
res = api_request(f'{DASHBOARD_API}/jobs', method='POST', payload=payload)
jid = res.get('id') or res.get('job_id')
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f'{DASHBOARD_API}/jobs/{jid}')
    if j.get('status') in ['succeeded', 'failed']:
        print(fetch_log(jid))
        break

