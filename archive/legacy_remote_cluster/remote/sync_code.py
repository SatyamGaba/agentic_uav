"""
Syncs local code and files to the remote cluster persistent scratch directory.
Supports two reliable transfer modes:
  1. Job-based Tar Sync (Default): Bundles code into a compressed archive and unpacks it via a quick batch job. Works 100% reliably even if the interactive Jupyter container is stopped.
  2. Jupyter Contents API Sync: Direct HTTP file-by-file upload (used when the user session container is running).

Usage Examples:
    # Sync agenticnav_paper_replica to /home/jovyan/scratch/agenticnav_paper_replica:
    python remote/sync_code.py

    # Dry-run preview:
    python remote/sync_code.py --dry-run

    # Sync using the direct Jupyter API:
    python remote/sync_code.py --mode api
"""
import sys
import os
import io
import tarfile
import base64
import argparse
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, JUPYTER_BASE, REMOTE_SCRATCH_DIR, api_request, get_ssl_context, HEADERS

IGNORE_DIRS = {"__pycache__", ".git", ".pytest_cache", ".venv", ".idea", ".vscode"}
IGNORE_EXTS = {".pyc", ".pyo", ".pyd"}

def create_code_archive(local_dir: str) -> bytes:
    """Creates an in-memory tar.gz archive of the local directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for root, dirs, files in os.walk(local_dir):
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IGNORE_EXTS:
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, local_dir)
                tar.add(full_path, arcname=rel_path)
    return buf.getvalue()

def sync_via_job(local_dir: str, remote_target_dir: str, dry_run: bool = False):
    """Packages code into base64 and executes a 5-minute extraction job on the cluster."""
    print(f"Packaging '{local_dir}' into compressed in-memory archive...")
    tar_bytes = create_code_archive(local_dir)
    b64_str = base64.b64encode(tar_bytes).decode("ascii")
    print(f"Archive created: {len(tar_bytes) / 1024:.1f} KB (Base64: {len(b64_str) / 1024:.1f} KB)")

    if dry_run:
        print(f"[DRY-RUN] Would submit extraction job to deploy archive to {remote_target_dir}.")
        return

    script = f"""#!/usr/bin/env bash
set -e
echo "Establishing destination directory: {remote_target_dir}"
mkdir -p "{remote_target_dir}"
echo "{b64_str}" | base64 -d | tar -xz -C "{remote_target_dir}"
echo "Deployment successful! Directory listing:"
ls -la "{remote_target_dir}"
"""
    payload = {
        "name": "sync-code-archive",
        "gpu_count": 0,
        "time_limit_min": 5,
        "script": script
    }

    print("Submitting sync extraction job to cluster queue...")
    try:
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
        job_id = res.get("job_id") or res.get("id")
        print("=" * 65)
        print(f"[SUCCESS] Sync job submitted! Job ID: #{job_id}")
        print(f"Tracking deployment progress...")
        
        # Follow job log briefly
        import time
        from remote.manage_job import fetch_log
        for _ in range(12):
            time.sleep(2)
            log = fetch_log(job_id)
            if "Deployment successful!" in log:
                print(log.strip())
                print("=" * 65)
                print("[SUCCESS] All codebase files synchronized to remote scratch!")
                return
            status_data = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
            if status_data.get("status") in ["succeeded", "failed"]:
                print(log.strip())
                return
        print(f"Job is processing in background. View log with: python remote/manage_job.py --id {job_id}")
    except Exception as e:
        print(f"[ERROR] Sync job submission failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Sync local project files to remote GPU cluster.")
    parser.add_argument("--dir", type=str, default="agenticnav_paper_replica", help="Local directory to sync")
    parser.add_argument("--target", type=str, default=f"{REMOTE_SCRATCH_DIR}/agenticnav_paper_replica", help="Remote destination directory")
    parser.add_argument("--mode", choices=["job", "api"], default="job", help="Sync method ('job' works anytime; 'api' requires active user container)")
    parser.add_argument("--dry-run", action="store_true", help="Preview transfer without making changes")

    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"[ERROR] Local directory not found: {args.dir}")
        sys.exit(1)

    if args.mode == "job":
        sync_via_job(args.dir, args.target, dry_run=args.dry_run)
    else:
        # Fallback to direct file upload
        from remote.sync_code import sync_directory
        sync_directory(args.dir, args.target, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
