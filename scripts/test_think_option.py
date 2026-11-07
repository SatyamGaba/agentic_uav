import urllib.request
import json
import time

def test_payload(name, payload_extra):
    payload = {
        'model': 'qwen3.5:4b',
        'messages': [{'role': 'user', 'content': 'What is 2+2? Answer in JSON {"result": 4}.'}],
        'stream': False,
        'options': {'temperature': 0.0}
    }
    payload.update(payload_extra)
    req = urllib.request.Request(
        'http://127.0.0.1:11434/api/chat',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        dt = time.time() - t0
        eval_count = data.get('eval_count', 0)
        content = data.get('message', {}).get('content', '')
        print(f"[{name}] Time: {dt:.2f}s, Tokens: {eval_count}, Reply: {content[:80]}")
    except Exception as e:
        print(f"[{name}] Failed: {e}")

test_payload("Baseline (no think flag)", {})
test_payload("think: false (top-level)", {"think": False})
test_payload("think: 'false' (top-level str)", {"think": "false"})
test_payload("options.think: false", {"options": {"temperature": 0.0, "think": False}})
test_payload("format: json", {"format": "json"})
test_payload("think: false + format: json", {"think": False, "format": "json"})

