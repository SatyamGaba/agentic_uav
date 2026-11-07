from config import DASHBOARD_API, api_request
from manage_job import fetch_log
import time

script = """#!/bin/bash
cd /home/jovyan/scratch/TEST/tool_alloc
python3 -m pytest tests -x --tb=short
"""

payload = {
    "name": "check-pytest-failure",
    "gpu_count": 1,
    "time_limit_min": 5,
    "script": script
}

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
job_id = res.get("id") or res.get("job_id")
print(f"Submitted Job #{job_id}")

for _ in range(40):
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
    if j.get("status") in ["succeeded", "failed"]:
        print(f"Job #{job_id} {j.get('status')}")
        print("--- LOG ---")
        print(fetch_log(job_id))
        break
