"""
Synchronizes all project files, code, and configurations to the remote GPU cluster.
Packages the codebase into chunks <= 45 KB to strictly respect the 65,536-byte API limit,
queues deployment on the cluster, and verifies file extraction.

Usage:
    python remote/sync_all.py
    python remote/sync_all.py --dry-run
"""
import sys
import os
import io
import tarfile
import base64
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, REMOTE_SCRATCH_DIR, api_request
from remote.manage_job import fetch_log

IGNORE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".venv", ".idea", ".vscode", "test_traces"}
IGNORE_EXTS = {".pyc", ".pyo", ".pyd"}

def gather_files(project_root: str):
    pkg_dir = os.path.join(project_root, "agenticnav_paper_replica")
    all_files = []
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in IGNORE_EXTS:
                continue
            all_files.append(os.path.join(root, f))
    v_script = os.path.join(project_root, "scratch", "verify_end_to_end_robustness.py")
    if os.path.exists(v_script):
        all_files.append(v_script)
    p_script = os.path.join(project_root, "scratch", "run_pilot_deep_inspection.py")
    if os.path.exists(p_script):
        all_files.append(p_script)
    ten_script = os.path.join(project_root, "scratch", "run_10_episodes_pilot.py")
    if os.path.exists(ten_script):
        all_files.append(ten_script)
    step_eval_script = os.path.join(project_root, "scratch", "run_pilot_step_by_step_eval.py")
    if os.path.exists(step_eval_script):
        all_files.append(step_eval_script)
    return all_files

def chunk_files(all_files, project_root, max_tar_bytes=28000):
    chunks = []
    current_chunk = []
    current_size = 0
    for f in all_files:
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            t.add(f, arcname=os.path.relpath(f, project_root))
        size = len(buf.getvalue())
        if current_size + size > max_tar_bytes and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [f]
            current_size = size
        else:
            current_chunk.append(f)
            current_size += size
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def main():
    parser = argparse.ArgumentParser(description="Synchronize all files to remote cluster.")
    parser.add_argument("--dry-run", action="store_true", help="Preview transfer archive size only.")
    parser.add_argument("--target-dir", type=str, default="/home/jovyan/scratch/agenticnav_paper_replica",
                        help="Remote destination path.")
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print("=" * 70)
    print("      SYNCHRONIZING CODEBASE TO REMOTE GPU CLUSTER SCRATCH        ")
    print("=" * 70)
    print(f"Local Root:  {project_root}")
    print(f"Destination: {args.target_dir}")

    all_files = gather_files(project_root)
    chunks = chunk_files(all_files, project_root, max_tar_bytes=28000)
    print(f"Total Files: {len(all_files)} split into {len(chunks)} safe payload chunks (< 65KB limit)")

    if args.dry_run:
        print("[DRY-RUN] File scan completed successfully. No network transfer performed.")
        return

    job_ids = []
    for i, chunk in enumerate(chunks):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for f in chunk:
                tar.add(f, arcname=os.path.relpath(f, project_root))
        b64_str = base64.b64encode(buf.getvalue()).decode("ascii")

        is_last = (i == len(chunks) - 1)
        verify_cmd = ""
        if is_last:
            verify_cmd = f"""
echo -e "\\nAll chunks unpacked! Validating deployment..."
cd "{args.target_dir}"
python3 -c "
import agenticnav_paper_replica
from agenticnav_paper_replica.geometry.pinhole import CameraConfig
print('Module import verification: SUCCESS')
"
echo "[DEPLOY_SUCCESS] Codebase fully deployed and verified!"
"""

        deploy_script = f"""#!/usr/bin/env bash
set -e
mkdir -p "{args.target_dir}"
mkdir -p "{args.target_dir}/runs"
mkdir -p "{args.target_dir}/scratch"
mkdir -p "{args.target_dir}/data/scene_datasets/mp3d"

echo "Extracting payload chunk {i+1}/{len(chunks)}..."
cat << 'EOF' | base64 -d | tar -xz -C "{args.target_dir}"
{b64_str}
EOF
echo "Chunk {i+1}/{len(chunks)} extracted."
{verify_cmd}
"""
        print(f"Submitting chunk {i+1}/{len(chunks)} (Script size: {len(deploy_script)} bytes)...")
        payload = {
            "name": f"setup-p{i+1}",
            "time_limit_min": 5,
            "script": deploy_script,
        }
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
        jid = res.get("id") or res.get("job_id")
        job_ids.append(jid)
        print(f"  -> Queued Job #{jid}")

    print("\n" + "=" * 70)
    print(f"[SUCCESS] All {len(job_ids)} sync jobs queued successfully: {job_ids}")
    print("=" * 70)

if __name__ == "__main__":
    main()
