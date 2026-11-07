import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = "ollama show qwen3-vl:8b --parameters && echo '---TEMPLATE---' && ollama show qwen3-vl:8b --template"
print(exec_cmd.run_remote(cmd, wait_sec=8.0))
