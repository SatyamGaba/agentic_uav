import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = "cat /home/jovyan/scratch/benchmark_40ep_live.log"
print(exec_cmd.run_remote(cmd, wait_sec=8.0))
