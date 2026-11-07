import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

status_cmd = """tail -n 25 /home/jovyan/scratch/benchmark_40ep_live.log"""
out = exec_cmd.run_remote(status_cmd, wait_sec=12.0)
print(out)
