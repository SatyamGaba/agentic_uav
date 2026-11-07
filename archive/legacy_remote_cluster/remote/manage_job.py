"""
Inspects, monitors, or cancels batch jobs on the Ananta GPU Cluster.

Usage Examples:
    # List the last 5 jobs:
    python remote/manage_job.py --list

    # View status and log of a specific job:
    python remote/manage_job.py --id 298

    # Tail/stream a running job until completion:
    python remote/manage_job.py --tail 316

    # Cancel a running or queued job:
    python remote/manage_job.py --cancel 316
"""
import sys
import os
import time
import argparse
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request, get_ssl_context, HEADERS

def fetch_log(job_id: int) -> str:
    """Fetches the console log for a given job."""
    import urllib.request
    url = f"{DASHBOARD_API}/jobs/{job_id}/log"
    req = urllib.request.Request(url, headers=HEADERS)
    ctx = get_ssl_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=30) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"[Log not available yet or HTTP {e.code}]"
    except Exception as e:
        return f"[Error fetching log: {e}]"

def list_jobs(limit: int = 10):
    res = api_request(f"{DASHBOARD_API}/jobs")
    jobs = res.get("jobs", [])
    if not jobs:
        print("No jobs found for user.")
        return

    print("=" * 80)
    print(f"{'ID':<6} {'NAME':<24} {'STATUS':<12} {'GPU':<5} {'DURATION':<10} {'SUBMITTED AT'}")
    print("-" * 80)
    for j in jobs[:limit]:
        jid = j.get("id")
        name = (j.get("name") or "unnamed")[:23]
        status = j.get("status") or "unknown"
        gpu = str(j.get("gpu_index")) if j.get("gpu_index") is not None else "-"
        elapsed = f"{j.get('elapsed_s'):.1f}s" if j.get("elapsed_s") else "-"
        sub_at = (j.get("submitted_at") or "")[:19].replace("T", " ")
        print(f"#{jid:<5} {name:<24} {status:<12} {gpu:<5} {elapsed:<10} {sub_at}")
    print("=" * 80)

def show_job(job_id: int):
    try:
        j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
    except Exception as e:
        print(f"[ERROR] Could not fetch job #{job_id}: {e}")
        return

    print("=" * 70)
    print(f"JOB DETAILS: #{job_id} ({j.get('name')})")
    print("=" * 70)
    print(f"Status:       {j.get('status')} (Exit code: {j.get('exit_code')}, End reason: {j.get('end_reason')})")
    print(f"GPU Index:    {j.get('gpu_index')} | Time Limit: {j.get('time_limit_min')}m")
    print(f"Elapsed:      {j.get('elapsed_s')}s")
    print(f"Submitted:    {j.get('submitted_at')}")
    print(f"Started:      {j.get('started_at')}")
    print(f"Ended:        {j.get('ended_at')}")
    print("-" * 70)
    print("LOG OUTPUT:")
    print("-" * 70)
    log_content = fetch_log(job_id)
    print(log_content.strip() if log_content.strip() else "(Empty log)")
    print("=" * 70)

def tail_job(job_id: int, poll_interval: int = 5):
    print(f"Streaming logs for Job #{job_id} (Ctrl+C to stop)...")
    last_log = ""
    while True:
        try:
            j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
            status = j.get("status")
            curr_log = fetch_log(job_id)
            
            # Print new chunk if any
            if curr_log != last_log:
                if curr_log.startswith(last_log):
                    print(curr_log[len(last_log):], end="", flush=True)
                else:
                    print(curr_log, flush=True)
                last_log = curr_log
            
            if status in ["succeeded", "failed", "cancelled", "killed"]:
                print(f"\n[JOB TERMINATED] Final status: {status} (exit code: {j.get('exit_code')})")
                break
        except KeyboardInterrupt:
            print("\nStopped watching log (job continues running in background).")
            break
        except Exception as e:
            print(f"[Warning: poll error: {e}]")
        time.sleep(poll_interval)

def cancel_job(job_id: int):
    print(f"Cancelling Job #{job_id}...")
    try:
        res = api_request(f"{DASHBOARD_API}/jobs/{job_id}/cancel", method="POST")
        print(f"[SUCCESS] Job #{job_id} cancellation request accepted: {res}")
    except Exception as e:
        print(f"[ERROR] Failed to cancel job #{job_id}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage and monitor remote GPU cluster jobs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List recent jobs")
    group.add_argument("--id", type=int, help="Show status and log for a specific Job ID")
    group.add_argument("--tail", type=int, help="Stream/follow log for a running Job ID")
    group.add_argument("--cancel", type=int, help="Cancel a queued or running Job ID")
    parser.add_argument("--limit", type=int, default=10, help="Number of jobs to list (default 10)")

    args = parser.parse_args()

    if args.list:
        list_jobs(args.limit)
    elif args.id:
        show_job(args.id)
    elif args.tail:
        tail_job(args.tail)
    elif args.cancel:
        cancel_job(args.cancel)

if __name__ == "__main__":
    main()

