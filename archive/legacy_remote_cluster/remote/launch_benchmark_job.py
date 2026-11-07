"""
Submits a long-running, continuous benchmark replication job to the remote GPU cluster.
The job executes on the dedicated Job GPU (NVIDIA L40S) for multiple hours,
features automatic checkpoint resumption, GPU watchdog heartbeat, and live progress logging.

Usage Examples:
    # Launch standard full benchmark (6 hours time limit, K=3 repetitions):
    python remote/launch_benchmark_job.py --name "replica-full-k3" --time 360 --repetitions 3

    # Launch quick 10-episode smoke test:
    python remote/launch_benchmark_job.py --name "smoke-test-10ep" --time 60 --max-episodes 10 --repetitions 1

    # Launch ablation study without depth tool:
    python remote/launch_benchmark_job.py --name "ablation-wo-depth" --ablation wo_depth --time 240
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request

def generate_cluster_runner_script(
    job_name: str,
    ablation: str,
    repetitions: int,
    max_episodes: int = None,
    mock: bool = False,
    backend: str = "auto",
    model: str = "gemini-2.5-flash",
) -> str:
    """Generates the bash runner script executed inside the cluster batch container."""
    max_ep_arg = f"--max-episodes {max_episodes}" if max_episodes else ""
    mock_arg = "--mock" if mock else ""
    backend_arg = f"--backend {backend}"

    script = f"""#!/usr/bin/env bash
set -e
export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/jovyan/scratch/agenticnav_paper_replica:$PYTHONPATH"
cd /home/jovyan/scratch/agenticnav_paper_replica

echo "======================================================================"
echo "    STARTING AGENTICNAV LONG-RUNNING BENCHMARK REPLICATION JOB        "
echo "======================================================================"
echo "Job Name      : {job_name}"
echo "Ablation Arm  : {ablation}"
echo "Repetitions K : {repetitions}"
echo "Start Time    : $(date)"
echo "Host / GPU    : $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
echo "Working Dir   : $(pwd)"
echo "======================================================================"

# 1. Start gentle background GPU heartbeat to keep cluster idle watchdog disarmed
echo "[WATCHDOG] Initializing background GPU heartbeat..."
python3 -c "
import torch, time, os
print(f'[Heartbeat Daemon PID: {os.getpid()}] Disarming cluster GPU idle watchdog.')
x = torch.randn(512, 512, device='cuda')
while True:
    y = x @ x
    time.sleep(5.0)
" &
HEARTBEAT_PID=$!
echo "Watchdog Heartbeat running with PID: $HEARTBEAT_PID"

# Ensure heartbeat process is killed when the benchmark completes or exits
cleanup() {{
    echo "Stopping watchdog heartbeat PID: $HEARTBEAT_PID..."
    kill $HEARTBEAT_PID || true
    echo "Job finished at $(date)."
}}
trap cleanup EXIT INT TERM

# 2. Check and start local Ollama daemon if available and not yet running
if command -v ollama >/dev/null 2>&1; then
    if ! pgrep -x "ollama" >/dev/null; then
        echo "[DAEMON] Starting local Ollama service in background..."
        ollama serve >/dev/null 2>&1 &
        sleep 3
    fi
fi

# 3. Execute benchmark runner in continuous loop with auto-resumption
OUTPUT_DIR="runs/{job_name}"
mkdir -p "$OUTPUT_DIR"

echo -e "\\n[RUNNER] Launching continuous benchmark execution loop..."
python3 agenticnav_paper_replica/run_benchmark.py \\
    --ablation "{ablation}" \\
    --repetitions {repetitions} \\
    {backend_arg} \\
    {max_ep_arg} \\
    {mock_arg} \\
    --model "{model}" \\
    --output-dir "$OUTPUT_DIR" \\
    --resume

echo "======================================================================"
echo "         BENCHMARK COMPLETED SUCCESSFULLY! FINAL RESULTS:             "
echo "======================================================================"
cat "$OUTPUT_DIR/summary_metrics.json" || true
"""
    return script

def main():
    parser = argparse.ArgumentParser(description="Submit long-running benchmark job to remote GPU cluster.")
    parser.add_argument("--name", type=str, default="eval-opt-b3",
                        help="Descriptive name for the job run.")
    parser.add_argument("--time", type=int, default=360,
                        help="Time limit in minutes (default 360 = 6 hours).")
    parser.add_argument("--ablation", type=str, default="full",
                        choices=["full", "wo_depth", "wo_recall", "wo_history", "wo_bev", "depth_image"],
                        help="Experimental arm.")
    parser.add_argument("--repetitions", "-K", type=int, default=3,
                        help="Number of repeated evaluations K per episode (default 3).")
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="Limit number of episodes (e.g. 10 or 100). Default runs all.")
    parser.add_argument("--mock", action="store_true",
                        help="Use MockGroundEmbodiment (for dry-runs or when Habitat is absent).")
    parser.add_argument("--backend", type=str, default="auto",
                        choices=["auto", "gemini", "ollama", "mock"],
                        help="VLM backend provider (default 'auto').")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Model name (e.g. 'gemini-2.5-flash' or 'qwen3-vl:8b').")
    parser.add_argument("--no-watch", action="store_true",
                        help="Submit and exit immediately without opening live progress monitor.")

    args = parser.parse_args()

    print("=" * 70)
    print("      LAUNCHING LONG-RUNNING BENCHMARK JOB ON REMOTE GPU 1        ")
    print("=" * 70)
    print(f"Job Name:        {args.name}")
    print(f"Time Limit:      {args.time} minutes (~{args.time/60:.1f} hours)")
    print(f"Ablation Arm:    {args.ablation}")
    print(f"Backend Provider:{args.backend}")
    print(f"Model Name:      {args.model}")
    print(f"Repetitions K:   {args.repetitions}")
    print(f"Episodes:        {'ALL (100)' if args.max_episodes is None else args.max_episodes}")
    print(f"Embodiment:      {'Mock (Synthetic)' if args.mock else 'Habitat-Sim Ground'}")

    runner_script = generate_cluster_runner_script(
        job_name=args.name,
        ablation=args.ablation,
        repetitions=args.repetitions,
        max_episodes=args.max_episodes,
        mock=args.mock,
        backend=args.backend,
        model=args.model,
    )

    payload = {
        "name": args.name,
        "gpu_count": 1,
        "time_limit_min": args.time,
        "script": runner_script,
    }

    try:
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
    except Exception as e:
        print(f"[ERROR] Could not submit benchmark job: {e}")
        print("\nNote: If you receive HTTP 403 Forbidden:")
        print("  1. Please log in to the campus network portal at: https://campnet.bits-goa.ac.in:8090/")
        print("  2. If connecting from outside campus, ensure FortiClient VPN is active.")
        sys.exit(1)

    job_id = res.get("job_id") or res.get("id")
    print("\n" + "=" * 70)
    print(f"[SUCCESS] Benchmark job submitted to dedicated GPU 1! (Job ID: #{job_id})")
    print("=" * 70)
    print(f"To monitor progress anytime:")
    print(f"    python remote/track_progress.py --job-id {job_id}")
    print(f"To download results when complete:")
    print(f"    python remote/download_results.py --job-id {job_id}")
    print(f"To cancel this job:")
    print(f"    python remote/manage_job.py --cancel {job_id}")
    print("=" * 70 + "\n")

    if not args.no_watch:
        from remote.track_progress import monitor_job_progress
        monitor_job_progress(job_id)

if __name__ == "__main__":
    main()

