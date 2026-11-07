import urllib.request
import json
import base64
import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buf).decode('utf-8')

payload = {
    'model': 'moondream',
    'messages': [
        {
            'role': 'user',
            'content': 'You are an object detector. Return JSON matching: {"selected_mark_id": 1, "confidence": 0.9, "reasoning": "detected"}',
            'images': [b64]
        }
    ],
    'format': 'json',
    'stream': False
}
req = urllib.request.Request(
    'http://localhost:11434/api/chat',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read())
        print('NATIVE OLLAMA FORMAT JSON RES:')
        print(res['message']['content'])
except Exception as e:
    print('ERROR:', e)
