import urllib.request
import json
import io
import base64
from PIL import Image, ImageDraw

def test_pt(tx, ty):
    im = Image.new('RGB', (640, 480), color=(20, 20, 20))
    d = ImageDraw.Draw(im)
    d.ellipse([tx - 8, ty - 8, tx + 8, ty + 8], fill=(255, 30, 30))
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    prompt = (
        "This image has size 640x480. There is a single bright red dot in the image.\n"
        "Return the normalized coordinates of the center of this red dot in [0.0, 1.0] range as JSON:\n"
        "{\"u\": <float between 0.0 and 1.0>, \"v\": <float between 0.0 and 1.0>}\n"
        "Where (0.0, 0.0) is the top-left corner and (1.0, 1.0) is the bottom-right corner.\n"
        "Do not include explanation, output only the JSON object."
    )
    payload = {
        'model': 'qwen3.5:4b',
        'messages': [{'role': 'user', 'content': prompt, 'images': [b64]}],
        'stream': False,
        'options': {'temperature': 0.0}
    }
    req = urllib.request.Request(
        'http://127.0.0.1:11434/api/chat',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    print(f"TRUTH: tx={tx}, ty={ty} -> u={tx/640:.4f}, v={ty/480:.4f}")
    print(f"DONE REASON: {res.get('done_reason')}")
    print("RAW REPLY:\n", res.get('message', {}).get('content', ''))

test_pt(320, 240)

