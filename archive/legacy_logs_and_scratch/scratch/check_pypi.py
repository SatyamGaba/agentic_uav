import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log
import time

script = """#!/usr/bin/env bash
python3 -c '
import urllib.request
print("Testing internet / PyPI connectivity...")
try:
    with urllib.request.urlopen("https://pypi.org", timeout=5) as r:
        print("PyPI reachable, status:", r.status)
except Exception as e:
    print("PyPI not reachable:", e)
'
"""

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={"name": "test-pypi", "time_limit_min": 5, "script": script})
jid = res.get("id")
print("Submitted pypi check job:", jid)
for _ in range(15):
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

