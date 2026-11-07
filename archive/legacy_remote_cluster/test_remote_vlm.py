import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = """python -c "import sys; sys.path.insert(0, '/home/jovyan/scratch'); from framework.backends.ollama_client import OllamaVLMBackend; b = OllamaVLMBackend(model_name='qwen3-vl:8b'); r = b.chat([{'role': 'user', 'content': 'List 3 numbers'}]); print('Response:', r.content, 'Latency:', r.latency_s)" """

print("Querying remote qwen3-vl:8b...")
res = exec_cmd.run_remote(cmd, wait_sec=15.0)
print(res)
