import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log
import time

script = """#!/usr/bin/env bash
/home/jovyan/scratch/conda_envs/openfly/bin/python -c '
import sys
print("OpenFly env Python:", sys.version)
for mod in ["airsim", "habitat_sim", "cv2", "torch", "numpy"]:
    try:
        __import__(mod)
        print(f"  {mod}: AVAILABLE")
    except ImportError as e:
        print(f"  {mod}: NOT available ({e})")
'
"""

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={"name": "check-openfly-env", "time_limit_min": 5, "script": script})
jid = res.get("id")
print("Submitted env check job:", jid)
for _ in range(15):
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

