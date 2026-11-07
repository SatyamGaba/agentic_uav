import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
cd /home/jovyan/scratch/TEST/tool_alloc
python3 -c "
import numpy as np
t = np.load('cache/pilot_gpu_0p5b/pilot_gpu_0p5b_tree.npz')
v_in_ok = t['v_in_ok']
v_call_ok = t['v_call_ok']
v_no_ok = t['v_no_ok']
print('Total nodes:', v_in_ok.size)
print('v_in_ok True:', v_in_ok.sum())
print('v_in_ok False:', (~v_in_ok).sum())
print('v_call_ok True:', v_call_ok.sum())
print('v_call_ok False:', (~v_call_ok).sum())
print('Nodes where v_in_ok is True but v_call_ok is False:', (v_in_ok & ~v_call_ok).sum())
print('Nodes where v_in_ok is False and v_call_ok is False:', (~v_in_ok & ~v_call_ok).sum())
"
"""

payload = {"name": "verify-tool-ok", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

