import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

launch_cmd = """cd /home/jovyan/scratch && pkill -f run_framework_benchmark.py || true; sleep 1; rm -f benchmark_40ep_live.log benchmark_40ep_qwen3vl_seen.json && nohup python run_framework_benchmark.py --adapter calibrated --backend ollama --model qwen3-vl:8b --split benchmark/seen.json --num-episodes 40 --output benchmark_40ep_qwen3vl_seen.json > benchmark_40ep_live.log 2>&1 & echo $!"""

print("Launching 40-episode real VLM benchmark in background on remote L40S server...")
out = exec_cmd.run_remote(launch_cmd, wait_sec=6.0)
print("Remote Output (PID):", out.strip())
