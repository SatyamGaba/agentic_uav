import ssl
import time
import urllib.request
import json
import sys

TOKEN = "b4bbbe0a02884d46af65379e7b74f337"
URL = "https://172.24.16.175/user/krutik/api"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(URL, headers={"Authorization": f"token {TOKEN}"})

print("Monitoring server container initialization...", flush=True)

for attempt in range(1, 61):
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=8.0) as r:
            if r.status == 200:
                print(f"[SUCCESS] Remote server is ONLINE after attempt {attempt}! (HTTP 200)", flush=True)
                sys.exit(0)
    except urllib.error.HTTPError as e:
        print(f"Attempt {attempt}: HTTP {e.code} ({e.reason})", flush=True)
    except Exception as e:
        print(f"Attempt {attempt}: {e}", flush=True)
    time.sleep(5.0)

print("[TIMEOUT] Server did not become ready within 5 minutes.", flush=True)
sys.exit(1)
