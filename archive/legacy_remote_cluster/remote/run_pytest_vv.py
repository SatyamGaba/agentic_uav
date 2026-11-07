import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request
from manage_job import fetch_log
import time

script = """#!/bin/bash
cd /home/jovyan/scratch/TEST/tool_alloc
python3 -m pytest tests/test_l40s.py -k test_regression_extras_off_is_byte_identical_to_pre_l40s_code -vv
"""

payload = {"name": "pytest-vv", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

