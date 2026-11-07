import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = """cd /home/jovyan/scratch && python run_framework_benchmark.py --adapter calibrated --backend ollama --model qwen3-vl:8b --split benchmark/seen.json --num-episodes 1 --output bench_test_1ep.json"""

print("Executing 1-episode real VLM benchmark test on remote L40S GPU...")
res = exec_cmd.run_remote(cmd, wait_sec=65.0)
print(res)
