import urllib.request
import json
import io
import base64
import random
import re
import time
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

def run_calibration(num_samples=50):
    random.seed(42)
    width, height = 640, 480
    errors_px = []
    samples = []
    
    t0 = time.time()
    for i in range(num_samples):
        tx = random.randint(40, width - 40)
        ty = random.randint(40, height - 40)
        
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
            "Where (0.0, 0.0) is the top-left corner and (1.0, 1.0) is the bottom-right corner."
        )
        payload = {
            'model': 'qwen3.5:4b',
            'messages': [{'role': 'user', 'content': prompt, 'images': [b64]}],
            'stream': False,
            'think': False,
            'format': 'json',
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
            errors_px.append(err)
            samples.append({"sample": i+1, "truth": [tx, ty], "pred": [round(px, 1), round(py, 1)], "error_px": round(err, 2)})
        else:
            errors_px.append(100.0)
            samples.append({"sample": i+1, "truth": [tx, ty], "error_px": 100.0, "parse_error": reply})
            
    total_time = time.time() - t0
    med_err = float(np.median(errors_px))
    p90_err = float(np.percentile(errors_px, 90))
    med_pct = (med_err / width) * 100.0
    p90_pct = (p90_err / width) * 100.0
    
    print(f"Total time for {num_samples} samples: {total_time:.1f}s ({total_time/num_samples:.2f}s/sample)")
    print(f"Median error: {med_err:.2f}px ({med_pct:.2f}% of width)")
    print(f"90th percentile error: {p90_err:.2f}px ({p90_pct:.2f}% of width)")
    
    return {
        "num_samples": num_samples,
        "total_time_s": round(total_time, 2),
        "mean_latency_s": round(total_time/num_samples, 3),
        "median_error_px": round(med_err, 2),
        "median_error_pct": round(med_pct, 2),
        "p90_error_px": round(p90_err, 2),
        "p90_error_pct": round(p90_pct, 2),
        "samples": samples
    }

if __name__ == "__main__":
    res = run_calibration(50)
    with open("audit/phase2/g26_calibration_50pts_fast.json", "w") as f:
        json.dump(res, f, indent=2)

