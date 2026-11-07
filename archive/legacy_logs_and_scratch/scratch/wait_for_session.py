import sys, os, time, urllib.request, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import BASE_URL, TOKEN, get_ssl_context, HEADERS, DASHBOARD_API

ctx = get_ssl_context()
for i in range(25):
    req = urllib.request.Request(f"{DASHBOARD_API}/session", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
            data = json.loads(r.read().decode())
            state = data.get("state")
            print(f"[{i+1}] Session state: {state}")
            if state == "running":
                print(f"[SUCCESS] Session is running! jupyter_api: {data.get('jupyter_api')}")
                break
    except Exception as e:
        print(f"[{i+1}] Error: {e}")
    time.sleep(2)

