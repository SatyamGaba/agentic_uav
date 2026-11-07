"""
Submits a background batch job to the Ananta GPU Cluster queue.
The job executes on the dedicated Job GPU (NVIDIA L40S) without requiring an open terminal.

Usage Examples:
    # Submit a quick inline command:
    python remote/submit_job.py --name "check-gpu" --cmd "nvidia-smi" --time 5

    # Submit a bash script file:
    python remote/submit_job.py --name "run-benchmark" --file path/to/run.sh --time 30

    # Submit a python script:
    python remote/submit_job.py --name "train-job" --file path/to/train.py --time 60
"""
import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request

def main():
    parser = argparse.ArgumentParser(description="Submit a batch job to the GPU cluster.")
    parser.add_argument("--name", type=str, required=True, help="Descriptive name for the job")
    parser.add_argument("--time", type=int, default=10, help="Time limit in minutes (minimum 5, default 10)")
    parser.add_argument("--file", type=str, default=None, help="Path to local script file (.sh or .py)")
    parser.add_argument("--cmd", type=str, default=None, help="Inline shell command string")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs requested (default 1)")
    
    args = parser.parse_args()

    if not args.file and not args.cmd:
        print("[ERROR] You must provide either --file <script_path> or --cmd <bash_string>")
        sys.exit(1)

    if args.time < 5:
        print("[WARNING] Cluster enforces a minimum time limit of 5 minutes. Setting --time 5.")
        args.time = 5

    # Formulate script payload
    if args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] Script file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r", encoding="utf-8") as f:
            content = f.read()
        
        # If python file, wrap in python execution
        if args.file.endswith(".py"):
            script_text = f"#!/usr/bin/env bash\nset -e\npython3 - << 'EOF'\n{content}\nEOF\n"
        else:
            script_text = content
    else:
        script_text = f"#!/usr/bin/env bash\nset -e\n{args.cmd}\n"

    payload = {
        "name": args.name,
        "gpu_count": args.gpus,
        "time_limit_min": args.time,
        "script": script_text
    }

    print(f"Submitting job '{args.name}' ({args.time} min limit, {args.gpus} GPU)...")
    try:
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
    except Exception as e:
        print(f"[ERROR] Failed to submit job: {e}")
        sys.exit(1)

    job_id = res.get("job_id") or res.get("id")
    print("=" * 65)
    print(f"[SUCCESS] Job submitted successfully! Assigned Job ID: #{job_id}")
    print(f"Check job progress with:")
    print(f"    python remote/manage_job.py --id {job_id}")
    print(f"To cancel this job if needed:")
    print(f"    python remote/manage_job.py --cancel {job_id}")
    print("=" * 65)

if __name__ == "__main__":
    main()

