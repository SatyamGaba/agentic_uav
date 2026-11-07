import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
export HF_HOME=/home/jovyan/scratch/hf_cache
python3 -c "
import os
os.environ['HF_HUB_OFFLINE'] = '1'
from transformers import AutoConfig
c = AutoConfig.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', revision='7ae557604adf67be50417f59c2c2f167def9a775')
print('c attrs with commit:', [a for a in dir(c) if 'commit' in a])
print('_commit_hash:', getattr(c, '_commit_hash', 'NOT_FOUND'))
print('dir(c):', [a for a in dir(c) if not a.startswith('__')][:20])
"
"""

payload = {"name": "test-commit-hash", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

