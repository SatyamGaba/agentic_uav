import urllib.request
import ssl
import json

TOKEN = "b4bbbe0a02884d46af65379e7b74f337"
JOBS_URL = "https://172.24.16.175/services/gpu-dashboard/api/jobs"
ctx = ssl._create_unverified_context()
headers = {"Authorization": f"token {TOKEN}", "Content-Type": "application/json"}

script = """
import os
path = "/home/jovyan/scratch/agenticnav_paper_replica"
os.makedirs(path, exist_ok=True)
print(f"Remote replica directory established at: {path}")
"""

payload = {
    "name": "setup_agenticnav_replica_dir",
    "gpu_count": 0,
    "time_limit_min": 5,
    "code": script
}

try:
    req = urllib.request.Request(JOBS_URL, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, context=ctx) as resp:
        res = json.loads(resp.read().decode())
        print(f"Directory creation job submitted successfully! Job ID: {res.get('job_id')}")
except Exception as e:
    print(f"Notice: Remote setup can run once VPN is connected: {e}")

