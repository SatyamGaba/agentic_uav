import urllib.request
import json
import io
import base64
import random
import re
import numpy as np
from PIL import Image, ImageDraw

def extract_coords(text):
    m_json = re.search(r'\{[^{}]*\}', text)
    if m_json:
        try:
            d = json.loads(m_json.group(0))
            if "u" in d and "v" in d:
                return float(d["u"]), float(d["v"])
        except Exception:
            pass
    m_u = re.search(r'\"?u\"?\s*[:=]\s*([0-9\.]+)', text)
    m_v = re.search(r'\"?v\"?\s*[:=]\s*([0-9\.]+)', text)
    if m_u and m_v:
        return float(m_u.group(1)), float(m_v.group(1))
    return None, None

def run_tests(num_points=10):
    random.seed(42)
    width, height = 640, 480
    errors = []
    
    for i in range(num_points):
        tx = random.randint(60, width - 60)
        ty = random.randint(60, height - 60)
        
        im = Image.new('RGB', (width, height), color=(20, 20, 20))
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
            "Do not include commentary, output only the JSON object."
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
        
        reply = res.get('message', {}).get('content', '')
        pu, pv = extract_coords(reply)
        if pu is not None and pv is not None:
            if pu > 1.0: pu /= 1000.0
            if pv > 1.0: pv /= 1000.0
            px = pu * width
            py = pv * height
            err = np.sqrt((px - tx)**2 + (py - ty)**2)
            errors.append(err)
            print(f"Sample {i+1:2d}: truth=({tx:3d}, {ty:3d}), pred=({px:5.1f}, {py:5.1f}), err={err:5.2f}px ({err/width*100:4.2f}%)")
        else:
            print(f"Sample {i+1:2d}: PARSE FAILED. Reply: {reply[:100]}")
            errors.append(100.0)
            
    print(f"\nMedian error: {np.median(errors):.2f}px ({np.median(errors)/width*100:.2f}%)")
    print(f"90th percentile: {np.percentile(errors, 90):.2f}px ({np.percentile(errors, 90)/width*100:.2f}%)")

run_tests(10)

