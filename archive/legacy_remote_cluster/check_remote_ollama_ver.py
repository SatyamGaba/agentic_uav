import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = "ollama --version"
print(exec_cmd.run_remote(cmd, wait_sec=8.0))
