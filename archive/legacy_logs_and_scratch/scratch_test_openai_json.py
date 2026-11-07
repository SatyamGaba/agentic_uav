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
            'content': [
                {'type': 'text', 'text': 'Return valid JSON matching: {"selected_mark_id": 1, "confidence": 0.95, "reasoning": "marked object"}'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
            ]
        }
    ],
    'response_format': {'type': 'json_object'},
    'temperature': 0.1
}

req = urllib.request.Request(
    'http://localhost:11434/v1/chat/completions',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
try:
    with urllib.request.urlopen(req, timeout=30) as resp:
        res = json.loads(resp.read())
        print('OPENAI COMPAT WITH RESPONSE FORMAT RES:')
        print(res['choices'][0]['message']['content'])
except Exception as e:
    print('ERROR:', e)
