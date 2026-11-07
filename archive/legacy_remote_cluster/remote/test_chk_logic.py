import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
export HF_HOME=/home/jovyan/scratch/hf_cache
python3 << 'EOF'
import sys, os
os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoConfig, AutoTokenizer
from pathlib import Path

mid = 'Qwen/Qwen2.5-0.5B-Instruct'
rev = '7ae557604adf67be50417f59c2c2f167def9a775'

c = AutoConfig.from_pretrained(mid, revision=rev)
t = c.get_text_config()
AutoTokenizer.from_pretrained(mid, revision=rev)

h = getattr(c, '_commit_hash', None)
if not h:
    hf_home = Path(os.environ.get('HF_HOME', str(Path.home() / '.cache/huggingface')))
    tag = mid.replace('/', '--')
    m_dir = hf_home / 'hub' / f'models--{tag}' / 'snapshots' / rev
    if m_dir.exists():
        h = rev

print(t.num_hidden_layers, t.hidden_size, h)
EOF
"""

payload = {"name": "test-chk-2", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break
