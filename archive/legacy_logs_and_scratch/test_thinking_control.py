import sys
import exec_cmd

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# Write test script directly to remote scratch/test_think.py
remote_code = '''import urllib.request
import json

tests = [
    ("A: System direct", {
        "model": "qwen3-vl:8b",
        "messages": [
            {"role": "system", "content": "You are a robot controller. Output only a JSON array of integers. Do not think or explain."},
            {"role": "user", "content": "Fly forward 18m and stop. Action codes: 9=forward 9m, 0=stop."}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128}
    }),
    ("B: Stop token </think>", {
        "model": "qwen3-vl:8b",
        "messages": [
            {"role": "user", "content": "Fly forward 18m and stop. Action codes: 9=forward 9m, 0=stop. Respond in JSON array format: [9, 9, 0]"}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128, "stop": ["</think>"]}
    }),
    ("C: Prompt with /no-think or formatting", {
        "model": "qwen3-vl:8b",
        "messages": [
            {"role": "user", "content": "/no-think\\nOutput JSON array [9, 9, 0] to fly forward 18m."}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 128}
    }),
    ("D: High num_predict 4096 (allow full thinking)", {
        "model": "qwen3-vl:8b",
        "messages": [
            {"role": "user", "content": "Fly forward 18m and stop. Action codes: 9=forward 9m, 0=stop. End with [9, 9, 0]"}
        ],
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 4096}
    })
]

for name, payload in tests:
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=data, headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
        msg = resp.get("message", {})
        print(f"{name} -> think len: {len(msg.get('thinking', ''))} | content: {repr(msg.get('content', '')[:60])} | done: {resp.get('done_reason')}")
    except Exception as e:
        print(f"{name} -> ERROR: {e}")
'''

cmd = f"cat << 'EOF' > /home/jovyan/scratch/test_think.py\n{remote_code}\nEOF\npython /home/jovyan/scratch/test_think.py"
print(exec_cmd.run_remote(cmd, wait_sec=40.0))
