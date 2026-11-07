"""
Downloads benchmark logs, evaluation metrics, and trajectory traces from the remote cluster.
Packages the requested remote files and unpacks them locally without requiring an active Jupyter session.

Usage Examples:
    # Download all evaluation traces and metrics from remote:
    python remote/download_results.py --remote scratch/agenticnav_paper_replica/traces --local traces/remote_benchmark

    # Download a specific log file:
    python remote/download_results.py --file scratch/agenticnav_paper_replica/benchmark.log --local remote_benchmark.log
"""
import sys
import os
import io
import time
import base64
import tarfile
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request
from remote.manage_job import fetch_log

START_MARKER = "=== ARCHIVE_START ==="
END_MARKER = "=== ARCHIVE_END ==="

def download_remote_dir(remote_dir: str, local_dest: str):
    print(f"Requesting remote packaging of '{remote_dir}'...")
    script = f"""#!/usr/bin/env bash
set -e
if [ ! -d "{remote_dir}" ]; then
    echo "ERROR: Directory {remote_dir} does not exist!"
    exit 1
fi
echo "{START_MARKER}"
tar -cz -C "{remote_dir}" . | base64
echo "{END_MARKER}"
"""
    payload = {
        "name": "download-results-bundle",
        "gpu_count": 1,
        "time_limit_min": 5,
        "script": script
    }

    try:
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
        job_id = res.get("job_id") or res.get("id")
        print(f"Packaging job submitted (Job #{job_id}). Waiting for bundle...")
        
        while True:
            time.sleep(3)
            j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
            if j.get("status") in ["succeeded", "failed"]:
                raw_log = fetch_log(job_id)
                break
        
        if END_MARKER not in raw_log or START_MARKER not in raw_log:
            print("[ERROR] Failed to retrieve archive bundle from job output.")
            print("Log snippet:\n", raw_log[:500])
            return

        # Extract base64 payload
        b64_content = raw_log.split(START_MARKER)[1].split(END_MARKER)[0].strip()
        tar_bytes = base64.b64decode(b64_content)
        
        os.makedirs(local_dest, exist_ok=True)
        buf = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=buf, mode="r:gz") as tar:
            tar.extractall(local_dest)
        
        print("=" * 65)
        print(f"[SUCCESS] Downloaded and extracted bundle to: {local_dest}")
        print("Extracted files:")
        for root, _, files in os.walk(local_dest):
            for f in files:
                print(f"  - {os.path.relpath(os.path.join(root, f), local_dest)}")
        print("=" * 65)
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Download results and traces from remote GPU cluster.")
    parser.add_argument("--remote", type=str, default="/home/jovyan/scratch/agenticnav_paper_replica/traces", help="Remote directory path to download")
    parser.add_argument("--local", type=str, default="traces/remote_download", help="Local directory to extract into")

    args = parser.parse_args()
    download_remote_dir(args.remote, args.local)

if __name__ == "__main__":
    main()

