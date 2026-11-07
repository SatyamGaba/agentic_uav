import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

cmd = """python -c "
import urllib.request, json
# Test 1: num_predict 2048 to see when thinking ends and content begins
data = json.dumps({
    'model': 'qwen3-vl:8b',
    'messages': [
        {'role': 'user', 'content': 'Instruction: Fly forward to the building and stop. Return ONLY a JSON array of integers, e.g. [9, 9, 0]. No markdown.'}
    ],
    'stream': False,
    'options': {'num_predict': 1500, 'temperature': 0.0}
}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:11434/api/chat', data=data, headers={'Content-Type': 'application/json'})
resp = urllib.request.urlopen(req).read().decode('utf-8')
d = json.loads(resp)
print('THINKING LEN:', len(d['message'].get('thinking', '')))
print('CONTENT:', d['message'].get('content', ''))
print('DONE REASON:', d.get('done_reason', ''))
"
"""

print(exec_cmd.run_remote(cmd, wait_sec=30.0))
