"""
Real-time Terminal Progress Dashboard and Monitor for Remote GPU Benchmark Jobs.
Continuously polls the cluster API, parses [PROGRESS] markers and JSON state,
and displays a live ASCII progress bar, GPU telemetry, running SR/SPL metrics, and episode history.

Usage Examples:
    # Auto-detect and track the currently running benchmark job:
    python remote/track_progress.py

    # Track a specific Job ID:
    python remote/track_progress.py --job-id 345

    # Customize poll frequency (default 5s):
    python remote/track_progress.py --job-id 345 --poll-interval 10
"""
import sys
import os
import time
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request
from remote.manage_job import fetch_log

def format_duration(seconds: float) -> str:
    """Formats seconds into human-readable Hh Mm format."""
    if seconds is None or seconds < 0:
        return "N/A"
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m"
    return f"{mins}m {secs:02d}s"

def make_progress_bar(fraction: float, length: int = 30) -> str:
    """Creates an ASCII progress bar."""
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(length * fraction))
    bar = "=" * (filled - 1) + ">" if filled > 0 else ""
    bar = bar.ljust(length, " ")
    return f"[{bar}]"

def find_active_job() -> int:
    """Finds the Job ID of the most recent running or queued job."""
    res = api_request(f"{DASHBOARD_API}/status")
    running = res.get("queue", {}).get("running", [])
    if running:
        return running[0]["id"]
    
    # Check job list for most recent
    jobs_res = api_request(f"{DASHBOARD_API}/jobs")
    jobs = jobs_res.get("jobs", [])
    if jobs:
        return jobs[0]["id"]
    return None

def monitor_job_progress(job_id: int, poll_interval: int = 5):
    """Main monitoring loop with dashboard rendering."""
    print(f"Connecting to live progress stream for Job #{job_id}...")
    print("Press Ctrl+C at any time to exit monitor (remote job will continue running).\n")

    progress_pattern = re.compile(
        r"\[PROGRESS\]\s+k=(\d+)/(\d+)\s+ep=(\d+)/(\d+)\s+\((\d+)/(\d+),\s*([\d\.]+)%\)\s*\|\s*SR=(\w+)\s*\|\s*SPL=([\d\.]+)\s*\|\s*NE=([\d\.]+)m\s*\|\s*Cat=(\w+)\s*\|\s*ETA=([\d\.]+)m"
    )

    last_log_len = 0
    recent_episodes = []
    consecutive_errs = 0

    while True:
        try:
            # 1. Fetch Job State
            j = api_request(f"{DASHBOARD_API}/jobs/{job_id}", timeout=10)
            status = j.get("status", "unknown")
            job_name = j.get("name", "unnamed")
            elapsed_s = j.get("elapsed_s", 0)
            limit_m = j.get("time_limit_min", 0)

            # 2. Fetch GPU telemetry
            cluster_status = api_request(f"{DASHBOARD_API}/status", timeout=10)
            gpu_info = cluster_status.get("gpus", [{}, {}])[1] if len(cluster_status.get("gpus", [])) > 1 else {}
            gpu_temp = gpu_info.get("temp_c", "N/A")
            gpu_power = gpu_info.get("power_w", 0.0)
            gpu_vram_used = (gpu_info.get("mem_used") or 0) / 1e9
            gpu_vram_tot = (gpu_info.get("mem_total") or 0) / 1e9
            gpu_util = gpu_info.get("util", 0)

            # 3. Fetch Log & parse progress
            log_text = fetch_log(job_id)
            lines = log_text.splitlines()

            # Parse all progress lines
            matches = []
            for line in lines:
                m = progress_pattern.search(line)
                if m:
                    matches.append(m)

            # Clear screen or print clean section
            print("\033[H\033[J", end="") if os.name != "nt" else os.system("cls")
            print("=" * 72)
            print("         ANANTA REMOTE GPU CLUSTER: LIVE BENCHMARK MONITOR       ")
            print("=" * 72)
            print(f"Job:       #{job_id} ({job_name}) | Status: {status.upper()}")
            print(f"GPU:       NVIDIA L40S | {gpu_temp}°C, {gpu_power:.1f}W | VRAM: {gpu_vram_used:.1f} / {gpu_vram_tot:.1f} GB | Compute: {gpu_util}%")
            print(f"Time:      Elapsed: {format_duration(elapsed_s)} / Limit: {limit_m}m")
            print("-" * 72)

            if matches:
                latest = matches[-1]
                cur_k, tot_k = int(latest.group(1)), int(latest.group(2))
                cur_ep, tot_ep = int(latest.group(3)), int(latest.group(4))
                done_runs, total_runs = int(latest.group(5)), int(latest.group(6))
                pct = float(latest.group(7))
                eta_m = float(latest.group(12))

                # Compute aggregate metrics from parsed runs
                sr_list = [1.0 if m.group(8) == "True" else 0.0 for m in matches]
                spl_list = [float(m.group(9)) for m in matches]
                ne_list = [float(m.group(10)) for m in matches]

                mean_sr = (sum(sr_list) / len(sr_list)) * 100.0 if sr_list else 0.0
                mean_spl = (sum(spl_list) / len(spl_list)) * 100.0 if spl_list else 0.0
                mean_ne = (sum(ne_list) / len(ne_list)) if ne_list else 0.0

                bar = make_progress_bar(pct / 100.0, length=28)
                print(f"Overall Progress: {bar} {pct:.1f}% ({done_runs}/{total_runs} runs)")
                print(f"Active Episode  : Repetition {cur_k}/{tot_k} | Episode {cur_ep}/{tot_ep}")
                print(f"Estimated ETA   : ~{format_duration(eta_m * 60)}")
                print("-" * 72)
                print("RUNNING REPLICATION METRICS:")
                print(f"  Success Rate (SR) : {mean_sr:.1f}%  (Target: 76.0%)")
                print(f"  Path Weight (SPL) : {mean_spl:.1f}% (Target: 66.5%)")
                print(f"  Navigation Error  : {mean_ne:.2f}m (Target: 3.48m)")
                print("-" * 72)
                print("RECENT FINISHED RUNS (Last 5):")
                for m in matches[-5:]:
                    k_i, ep_i, sr_i, spl_i, ne_i, cat_i = m.group(1), m.group(3), m.group(8), m.group(9), m.group(10), m.group(11)
                    st_str = "SUCCESS" if sr_i == "True" else "FAILED "
                    print(f"  - Rep {k_i} Ep {int(ep_i):03d}: [{st_str}] SPL={float(spl_i):.3f} | NE={float(ne_i):.2f}m | Category={cat_i}")
            else:
                print("Initializing benchmark and loading dataset... No episode records yet.")
                print("\nLast 5 log lines:")
                for l in lines[-5:]:
                    if l.strip():
                        print(f"  {l}")

            print("=" * 72)
            print(f"(Polling every {poll_interval}s. Press Ctrl+C to stop watching)")

            if status in ["succeeded", "failed", "cancelled", "killed"]:
                print(f"\n[JOB TERMINATED] Final status: {status} (Exit Code: {j.get('exit_code')})")
                break

            consecutive_errs = 0

        except KeyboardInterrupt:
            print("\n[INFO] Monitor paused. The benchmark job is running in the background on GPU 1.")
            print(f"To resume monitoring at any time: python remote/track_progress.py --job-id {job_id}")
            break
        except Exception as e:
            consecutive_errs += 1
            if consecutive_errs > 10:
                print(f"\n[ERROR] Connection lost after 10 attempts: {e}")
                break
            time.sleep(2)

        time.sleep(poll_interval)

def main():
    parser = argparse.ArgumentParser(description="Live monitor for remote benchmark jobs.")
    parser.add_argument("--job-id", type=int, default=None,
                        help="Specific Job ID to monitor. If omitted, auto-detects active job.")
    parser.add_argument("--poll-interval", type=int, default=5,
                        help="Polling interval in seconds (default 5).")
    args = parser.parse_args()

    job_id = args.job_id
    if job_id is None:
        try:
            job_id = find_active_job()
        except Exception as e:
            print(f"[ERROR] Could not query active jobs: {e}")
            print("\nNote: If you receive HTTP 403 Forbidden:")
            print("  1. Please log in to the campus network portal at: https://campnet.bits-goa.ac.in:8090/")
            print("  2. If connecting from outside campus, ensure FortiClient VPN is active.")
            sys.exit(1)

    if job_id is None:
        print("[INFO] No active jobs found running on the cluster.")
        sys.exit(0)

    monitor_job_progress(job_id, poll_interval=args.poll_interval)

if __name__ == "__main__":
    main()

