import sys
import time
from pathlib import Path

sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request
from manage_job import fetch_log

def main():
    script = """#!/bin/bash
set -e
echo "=== Starting transfer from local host ==="
curl -f -s http://172.20.10.2:8765/Team7_Task4_L40S_package_v2.zip -o /home/jovyan/scratch/v2.zip
echo "Downloaded package:"
ls -lh /home/jovyan/scratch/v2.zip

echo "Extracting to /home/jovyan/scratch/TEST_V2..."
rm -rf /home/jovyan/scratch/TEST_V2 /home/jovyan/scratch/TEST_V2_TMP
mkdir -p /home/jovyan/scratch/TEST_V2 /home/jovyan/scratch/TEST_V2_TMP
unzip -q /home/jovyan/scratch/v2.zip -d /home/jovyan/scratch/TEST_V2_TMP

if [ -d /home/jovyan/scratch/TEST_V2_TMP/Team7_Task4_L40S_package_v2 ]; then
    mv /home/jovyan/scratch/TEST_V2_TMP/Team7_Task4_L40S_package_v2/* /home/jovyan/scratch/TEST_V2/
else
    mv /home/jovyan/scratch/TEST_V2_TMP/* /home/jovyan/scratch/TEST_V2/
fi
rm -rf /home/jovyan/scratch/TEST_V2_TMP /home/jovyan/scratch/v2.zip

echo "=== Unpack complete. Checking directory ==="
cd /home/jovyan/scratch/TEST_V2
ls -la

echo "=== Running sha256sum -c CHECKSUMS.sha256 ==="
sha256sum -c CHECKSUMS.sha256 | grep -v 'OK' || echo "ALL 78 CHECKSUMS OK!"
"""
    payload = {
        "name": "deploy-v2-package",
        "gpu_count": 0,
        "time_limit_min": 5,
        "script": script
    }
    res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
    jid = res.get("id") or res.get("job_id")
    print(f"Submitted deployment job #{jid}")

    for _ in range(25):
        time.sleep(2)
        s = api_request(f"{DASHBOARD_API}/jobs/{jid}")
        st = s.get("status")
        if st in ("succeeded", "failed"):
            print(f"Job #{jid} ended with status: {st}")
            print(fetch_log(jid))
            break

if __name__ == "__main__":
    main()

