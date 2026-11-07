import urllib.request
import json
import base64
import cv2
import numpy as np

img = np.zeros((360, 640, 3), dtype=np.uint8)
cv2.rectangle(img, (100, 100), (220, 240), (40, 180, 40), -1)
cv2.putText(img, '[1] water tank', (100, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
_, buf = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buf).decode('utf-8')

payload = {
    'model': 'moondream',
    'messages': [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': 'You are a drone vision navigator. Look at the image with candidate [1] water tank. Output valid JSON: {"selected_mark_id": 1, "confidence": 0.95, "reasoning": "Found water tank at mark 1"}'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}}
            ]
        }
    ],
    'temperature': 0.1
}

req = urllib.request.Request(
    'http://localhost:11434/v1/chat/completions',
    data=json.dumps(payload).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)
with urllib.request.urlopen(req, timeout=30) as resp:
    res = json.loads(resp.read())
    raw = res['choices'][0]['message']['content']
    print('RAW RESPONSE:')
    print(repr(raw))
    print('---')
    print(raw)
