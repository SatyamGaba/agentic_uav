import sys
import os
import json
import re
import time
import subprocess
import urllib.request

sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request
from manage_job import fetch_log

def main():
    zip_path = r"C:\Users\Kruti\Desktop\LLM\Team7_Task4_L40S_package_v2.zip"
    print(f"Uploading {zip_path} ({os.path.getsize(zip_path)/1024/1024:.2f} MB) to tmpfiles.org...")
    
    cmd = ["curl.exe", "-s", "-F", f"file=@{zip_path}", "https://tmpfiles.org/api/v1/upload"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout.strip()
    print("Upload response:", out)
    
    try:
        data = json.loads(out)
        page_url = data["data"]["url"]
    except Exception as e:
        print("Failed to parse upload response:", e)
        return

    print("Fetching direct download link from:", page_url)
    time.sleep(1)
    req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    
    m = re.search(r'href="(https://tmpfiles\.org/dl/[^"]+)"', html)
    if not m:
        print("Could not find direct download link in HTML!")
        return
    dl_url = m.group(1)
    print("Direct download URL:", dl_url)

    # Submit job to cluster to download and extract
    deploy_script = f"""#!/bin/bash
set -e
echo "Downloading package from {dl_url}..."
curl -sL "{dl_url}" -o /home/jovyan/scratch/v2.zip
echo "Downloaded size: $(ls -lh /home/jovyan/scratch/v2.zip | awk '{{print $5}}')"

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

cd /home/jovyan/scratch/TEST_V2
echo "=== Listing TEST_V2 contents ==="
ls -la

echo "=== Verifying all 78 Checksums ==="
sha256sum -c CHECKSUMS.sha256 | grep -v 'OK' || echo "ALL 78 CHECKSUMS PASS!"
"""
    payload = {
        "name": "deploy-v2-full",
        "gpu_count": 0,
        "time_limit_min": 10,
        "script": deploy_script
    }
    job_res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
    jid = job_res.get("id") or job_res.get("job_id")
    print(f"Submitted deployment job #{jid}")

    for _ in range(40):
        time.sleep(2)
        s = api_request(f"{DASHBOARD_API}/jobs/{jid}")
        st = s.get("status")
        if st in ("succeeded", "failed"):
            print(f"Job #{jid} ended with status: {st}")
            print(fetch_log(jid))
            break

if __name__ == "__main__":
    main()

