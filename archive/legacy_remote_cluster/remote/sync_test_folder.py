import os
import io
import tarfile
import base64
import time
import sys

from config import DASHBOARD_API, api_request
from manage_job import fetch_log

def main():
    pkg_dir = "C:\\Users\\Kruti\\Desktop\\LLM\\Team7_Task4_L40S_package\\Team7_Task4_L40S_package"
    target_dir = "/home/jovyan/scratch/TEST"

    print(f"Creating compact code archive from: {pkg_dir}")
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(pkg_dir):
            rel_root = os.path.relpath(root, pkg_dir).replace("\\", "/")
            if "cache/primary" in rel_root or "__pycache__" in rel_root:
                continue
            for f in files:
                if f.endswith((".pyc", ".zip")):
                    continue
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, pkg_dir)
                tar.add(full_path, arcname=rel_path)

    tar_bytes = buf.getvalue()
    b64_str = base64.b64encode(tar_bytes).decode("ascii")
    print(f"Archive ready: {len(tar_bytes) / 1024:.1f} KB (Base64: {len(b64_str) / 1024:.1f} KB)")

    CHUNK_SIZE = 40000  # 40 KB per chunk (script limit is 65,536 bytes)
    chunks = [b64_str[i:i+CHUNK_SIZE] for i in range(0, len(b64_str), CHUNK_SIZE)]
    num_chunks = len(chunks)
    print(f"Splitting into {num_chunks} chunks of <= {CHUNK_SIZE} bytes...")

    # Ensure remote directory is clean
    init_cmd = f"#!/bin/bash\nset -e\nmkdir -p '{target_dir}'\nrm -f '{target_dir}/pkg.b64'\n"
    res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload={"name": "init-test-dir", "gpu_count": 1, "time_limit_min": 5, "script": init_cmd})
    init_id = res.get("id") or res.get("job_id")
    print(f"Cleaned target directory via Job #{init_id}")
    
    # Wait for init
    for _ in range(15):
        time.sleep(1)
        j = api_request(f"{DASHBOARD_API}/jobs/{init_id}")
        if j.get("status") in ["succeeded", "failed"]:
            break

    # Upload chunks sequentially
    for idx, chunk in enumerate(chunks):
        is_last = (idx == num_chunks - 1)
        redirect = ">>"
        
        script = f"#!/bin/bash\nset -e\ncat << 'EOF' {redirect} '{target_dir}/pkg.b64'\n{chunk}\nEOF\n"
        if is_last:
            script += f"""
echo "Unpacking package in {target_dir}..."
cd '{target_dir}'
base64 -d '{target_dir}/pkg.b64' | tar -xz
rm -f '{target_dir}/pkg.b64'
echo "=== EXTRACTION SUCCESSFUL ==="
ls -lah '{target_dir}'
"""
        job_payload = {
            "name": f"upload-chunk-{idx+1}-of-{num_chunks}",
            "gpu_count": 1,
            "time_limit_min": 5,
            "script": script
        }
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=job_payload)
        job_id = res.get("id") or res.get("job_id")
        print(f"Uploaded chunk {idx+1}/{num_chunks} (Job #{job_id})")

        # Wait for chunk to append
        for _ in range(20):
            time.sleep(1)
            j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
            if j.get("status") in ["succeeded", "failed"]:
                if is_last:
                    print("--- FINAL EXTRACTION LOG ---")
                    print(fetch_log(job_id))
                break

    print("=" * 65)
    print(f"[SUCCESS] Code package deployed to remote server at: {target_dir}")
    print("=" * 65)

if __name__ == "__main__":
    main()
