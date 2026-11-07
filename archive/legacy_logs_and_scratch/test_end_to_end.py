import urllib.request
import json
import time

def test_endpoint(port, name):
    url = f"http://127.0.0.1:{port}/command"
    payload = {
        "prompt": "move towards the manor and focus on its top left window",
        "api_provider": "lmstudio",
        "api_key": "http://127.0.0.1:1234/v1/chat/completions",
        "model_name": "qwen/qwen3-vl-4b"
    }
    print(f"\n--- Testing {name} on port {port} ---")
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            print(f"Status: {resp.status}")
            print(f"Body: {resp.read().decode('utf-8')}")
            
        # Check telemetry
        tel_url = f"http://127.0.0.1:{port}/telemetry"
        with urllib.request.urlopen(tel_url, timeout=5) as resp:
            tel = json.loads(resp.read().decode('utf-8'))
            print(f"Telemetry state: {tel.get('state')}")
            print(f"VLM logs count: {len(tel.get('vlm_logs', []))}")
    except Exception as e:
        print(f"FAILED on port {port}: {e}")

if __name__ == "__main__":
    test_endpoint(8080, "MavicVLM Controller")
    test_endpoint(8081, "MultiUAV-Plat Controller")
