import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log
import time

script = """#!/usr/bin/env bash
python3 -m pip install --dry-run habitat-sim 2>&1 | head -n 20
"""

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={"name": "test-hab-pip", "time_limit_min": 5, "script": script})
jid = res.get("id")
print("Submitted pip check job:", jid)
for _ in range(15):
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

