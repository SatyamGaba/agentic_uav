import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = """python -c "
import urllib.request, json
data = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [
        {'role': 'system', 'content': 'You are a precise autonomous drone navigation planner. Output strictly valid JSON arrays of action integers. No pleasantries, no conversational preamble.'},
        {'role': 'user', 'content': 'Instruction: Fly forward to the building and stop.\\nReturn ONLY a JSON array of integers, e.g. [9, 9, 0]'}
    ],
    'stream': False,
    'options': {'num_predict': 512, 'temperature': 0.0}
}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req).read().decode('utf-8')
print(resp)
"
"""

print(exec_cmd.run_remote(cmd, wait_sec=20.0))
