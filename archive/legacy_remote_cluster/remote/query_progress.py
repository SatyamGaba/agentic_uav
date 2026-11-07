import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, HEADERS, get_ssl_context, api_request
import urllib.request
import json
import time

def get_job_log(job_id, max_chars=4000):
    try:
        req = urllib.request.Request(f"{DASHBOARD_API}/jobs/{job_id}/log", headers=HEADERS)
        with urllib.request.urlopen(req, context=get_ssl_context(), timeout=5) as resp:
            text = resp.read().decode("utf-8", errors="replace")
            return text[-max_chars:] if len(text) > max_chars else text
    except Exception as e:
        return f"[Error fetching log: {e}]"

def query_progress():
    print("=" * 75)
    print("                 L40S CLUSTER LIVE PROGRESS MONITOR")
    print("=" * 75)
    
    try:
        status = api_request(f"{DASHBOARD_API}/status")
    except Exception as e:
        print(f"Error connecting to cluster: {e}")
        return

    job_gpu = status.get("gpus", [{}, {}])[1]
    running_jobs = status.get("queue", {}).get("running", [])
    
    print(f"GPU 1 (Job GPU) : {job_gpu.get('name', 'NVIDIA L40S')}")
    print(f"  Utilization   : {job_gpu.get('util', 0)}%")
    print(f"  VRAM Memory   : {job_gpu.get('mem_used', 0) / 1e9:.2f} GB / {job_gpu.get('mem_total', 0) / 1e9:.2f} GB")
    print(f"  Temperature   : {job_gpu.get('temp_c', 0)} C | Power: {job_gpu.get('power_w', 0):.1f} W")
    print(f"  System Load   : {status.get('system', {}).get('load', [])}")
    
    if not running_jobs:
        print("\n[Status: IDLE] No job is currently executing on the Job GPU.")
        try:
            jobs_resp = api_request(f"{DASHBOARD_API}/jobs")
            jobs = jobs_resp.get("jobs", []) if isinstance(jobs_resp, dict) else jobs_resp
            if jobs:
                last_j = jobs[0]
                print(f"\nLast Executed Job:")
                print(f"  Job ID       : #{last_j['id']} ({last_j.get('name')})")
                print(f"  Status       : {last_j.get('status').upper()} (Exit Code: {last_j.get('exit_code')})")
                print(f"  End Reason   : {last_j.get('end_reason')}")
                print(f"  Elapsed Time : {last_j.get('elapsed_s', 0)/60:.1f} minutes")
                print("\n--- Recent Log Output ---")
                log = get_job_log(last_j['id'], 1500)
                lines = [l for l in log.strip().splitlines() if l.strip()]
                for line in lines[-12:]:
                    print(f"  {line}")
        except Exception as e:
            print(f"Error reading job history: {e}")
        print("=" * 75)
        return

    job = running_jobs[0]
    jid = job["id"]
    elapsed_m = job.get("elapsed_s", 0) / 60
    limit_m = job.get("limit_s", 0) / 60
    remaining_m = max(0, limit_m - elapsed_m)
    idle_s = job.get("gpu_idle_s", 0)
    cpu = job.get("cpu_percent", 0)
    mem_gb = job.get("rss_bytes", 0) / 1e9

    print(f"\nActive Job: #{jid} [{job.get('name')}]")
    print(f"  Status        : {job.get('status').upper()}")
    print(f"  Elapsed Time  : {elapsed_m:.1f} min / {limit_m:.0f} min (Remaining: {remaining_m:.1f} min)")
    print(f"  Idle Timer    : {idle_s:.1f}s (Watchdog: {'DISARMED' if idle_s < 10 else 'ACTIVE'})")
    print(f"  Container CPU : {cpu:.1f}% ({cpu/100:.1f} cores) | RSS Memory: {mem_gb:.2f} GB")

    log = get_job_log(jid, 3000)
    lines = [l for l in log.strip().splitlines() if l.strip()]
    
    # Check for progress indicators in log
    current_part = "1/3"
    current_step = "Starting..."
    headroom = None
    for line in reversed(lines):
        if "Partition" in line and "/" in line:
            parts = line.split("Partition")
            if len(parts) > 1:
                current_part = parts[1].split()[0]
        if "Step" in line:
            current_step = line.split("Step")[-1].strip()
            break
        if "Headroom:" in line:
            headroom = line.split("Headroom:")[-1].strip().split()[0]

    # Visual progress bar based on estimated stages
    part_num = int(current_part[0]) if current_part[0].isdigit() else 1
    pct = min(99.0, (part_num - 1) * 33.3 + (15.0 if "Step" in current_step else 5.0))
    bar_len = 30
    filled = int(bar_len * (pct / 100))
    bar = "=" * filled + "-" * (bar_len - filled)

    print(f"\nEstimated Progress : [{bar}] {pct:.1f}%")
    print(f"  Current Partition: {current_part} | Sub-step: {current_step}")
    if headroom:
        print(f"  Gate Headroom    : {headroom}")

    print("\n--- Recent Live Log Output (Last 15 lines) ---")
    for line in lines[-15:]:
        safe_line = line.encode('ascii', errors='replace').decode('ascii')
        print(f"  {safe_line}")
    print("=" * 75)

if __name__ == "__main__":
    query_progress()
