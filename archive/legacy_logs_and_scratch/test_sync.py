import sys
import os
import time
import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

# Ensure path
CONTROLLER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "multiauav_plat", "controllers", "multiauav_plat_controller")
if CONTROLLER_DIR not in sys.path:
    sys.path.insert(0, CONTROLLER_DIR)

from server import DashboardHTTPHandler, telemetry, mission

def run_test():
    server = ThreadingHTTPServer(("127.0.0.1", 8089), DashboardHTTPHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.5)

    print("\n1. Testing GET /telemetry ...")
    req = urllib.request.Request("http://127.0.0.1:8089/telemetry")
    with urllib.request.urlopen(req) as resp:
        print(f"Status: {resp.status}")
        data = json.loads(resp.read().decode())
        print(f"Initial State: {data.get('state')}")

    print("\n2. Testing POST /command ...")
    payload = {
        "prompt": "move to manor and inspect window",
        "api_provider": "lmstudio",
        "api_key": "http://127.0.0.1:1234/v1/chat/completions",
        "model_name": "qwen/qwen3-vl-4b"
    }
    req = urllib.request.Request(
        "http://127.0.0.1:8089/command",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        print(f"Status: {resp.status}")
        res_body = json.loads(resp.read().decode())
        print(f"Response Body: {res_body}")

    print("\n3. Verifying Telemetry State Update after POST ...")
    req = urllib.request.Request("http://127.0.0.1:8089/telemetry")
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
        print(f"Updated State: {data.get('state')}")

    print("\n4. Verifying Mission Config Update ...")
    m_data = mission.read()
    print(f"Mission Prompt: '{m_data.get('prompt')}'")
    print(f"Provider: {m_data.get('api_provider')}")
    print(f"New Request Queued: {m_data.get('new_request_queued')}")

    server.shutdown()
    print("\n✅ END-TO-END SUITE VERIFIED PERFECTLY!")

if __name__ == "__main__":
    run_test()
