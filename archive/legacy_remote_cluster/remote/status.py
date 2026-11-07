"""
Checks real-time GPU utilization, system memory, daily quota, and job queue status on the remote cluster.

Usage:
    python remote/status.py
"""
import sys
import os

# Allow running directly from root or from within remote/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request

def format_bytes(b: int) -> str:
    """Formats bytes into human readable GB/MB."""
    if b is None:
        return "N/A"
    gb = b / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    return f"{b / (1024 ** 2):.1f} MB"

def main():
    try:
        data = api_request(f"{DASHBOARD_API}/status")
    except Exception as e:
        print(f"[ERROR] Failed to query cluster status: {e}")
        print("Tip: Ensure you are connected to the campus VPN or internal network.")
        sys.exit(1)

    print("=" * 70)
    print("                 ANANTA REMOTE GPU CLUSTER STATUS                     ")
    print("=" * 70)

    # 1. User & Budget
    me = data.get("me", {})
    budget = me.get("budget", {})
    rem_s = budget.get("remaining_s", 0)
    rem_h = rem_s / 3600.0 if rem_s else 0.0
    print(f"User:           {me.get('username')} (Access valid until: {me.get('access_until')})")
    print(f"Session State:  {me.get('session_state')}")
    print(f"Daily Budget:   {rem_s}s remaining (~{rem_h:.2f} hours)")
    print(f"Scratch Used:   {format_bytes(me.get('scratch', {}).get('used'))}")
    print("-" * 70)

    # 2. GPUs
    print("ACCELERATORS (NVIDIA L40S PCIe):")
    for gpu in data.get("gpus", []):
        idx = gpu.get("index")
        name = gpu.get("name")
        util = gpu.get("util", 0)
        used_mem = format_bytes(gpu.get("mem_used"))
        total_mem = format_bytes(gpu.get("mem_total"))
        temp = gpu.get("temp_c", 0)
        power = gpu.get("power_w", 0.0)
        sessions = ", ".join(gpu.get("sessions") or ["None"])
        job = gpu.get("job") or "None"
        role = gpu.get("role")
        print(f"  [GPU {idx} | Role: {role.upper()}] {name} ({temp}°C, {power:.1f}W)")
        print(f"     VRAM: {used_mem} / {total_mem} | Compute Util: {util}%")
        print(f"     Sessions: {sessions} | Active Job: {job}")
    print("-" * 70)

    # 3. System Compute & RAM
    sys_info = data.get("system", {})
    mem_used = format_bytes(sys_info.get("mem_used"))
    mem_total = format_bytes(sys_info.get("mem_total"))
    load = sys_info.get("load", [])
    load_str = ", ".join(f"{x:.1f}" for x in load) if load else "N/A"
    print("SYSTEM COMPUTE:")
    print(f"  CPU Cores:    {sys_info.get('cores')} hardware threads (CPU Util: {sys_info.get('cpu_percent')}%)")
    print(f"  Load Average: {load_str}")
    print(f"  RAM Memory:   {mem_used} / {mem_total}")
    print("-" * 70)

    # 4. Job Queue
    queue = data.get("queue", {})
    running = queue.get("running", [])
    queued = queue.get("queued", [])
    print(f"JOB QUEUE: {len(running)} Running, {len(queued)} Queued")
    if running:
        for j in running:
            print(f"  -> RUNNING: Job #{j.get('id')} ({j.get('name')}) by {j.get('user')} on GPU {j.get('gpu_index')}")
    if queued:
        for j in queued:
            print(f"  -> QUEUED : Job #{j.get('id')} ({j.get('name')}) by {j.get('user')} (Pos: {j.get('position')})")
    if not running and not queued:
        print("  -> (Queue is currently empty. GPU 1 is idle and available for jobs!)")
    print("=" * 70)

if __name__ == "__main__":
    main()

