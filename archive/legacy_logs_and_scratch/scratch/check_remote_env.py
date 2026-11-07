import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import api_request, DASHBOARD_API
from remote.manage_job import fetch_log
import time

script = """#!/usr/bin/env bash
python3 - << 'PYEOF'
import sys, os

print("Python version:", sys.version)

try:
    import habitat_sim
    print("habitat_sim version:", habitat_sim.__version__)
except ImportError as e:
    print("habitat_sim NOT found:", e)

# Search for mp3d / scene files
search_paths = [
    "/home/jovyan/scratch",
    "/data",
    "/datasets",
    "/srv/iotlab",
    "/home/jovyan"
]
for p in search_paths:
    if os.path.exists(p):
        print(f"Scanning {p}...")
        try:
            for root, dirs, files in os.walk(p):
                glb_files = [f for f in files if f.endswith(".glb")]
                if glb_files:
                    print(f"  Found {len(glb_files)} .glb files in {root}: {glb_files[:3]}")
                if "mp3d" in root.lower() or "matterport" in root.lower():
                    print(f"  Found MP3D dir: {root}")
                # limit depth
                if root.count(os.sep) - p.count(os.sep) > 3:
                    dirs.clear()
        except Exception as err:
            print(f"  Error scanning {p}: {err}")
PYEOF
"""

res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={"name": "check-env-data", "time_limit_min": 5, "script": script})
jid = res.get("id")
print("Submitted env check job:", jid)
for _ in range(15):
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

