"""
Central configuration for interacting with the Ananta Remote GPU Cluster.
All remote scripts import configuration and helper sessions from this module.
"""
import ssl
import json
import urllib.request

# Cluster Endpoints
SERVER_IP = "172.24.16.175"
BASE_URL = f"https://{SERVER_IP}"
DASHBOARD_API = f"{BASE_URL}/services/gpu-dashboard/api"
USER = "krutik"
JUPYTER_BASE = f"{BASE_URL}/user/{USER}"
WS_BASE = f"wss://{SERVER_IP}/user/{USER}"

# API Authentication Token
TOKEN = "b4bbbe0a02884d46af65379e7b74f337"

# Persistent Remote Directories
REMOTE_SCRATCH_DIR = "/home/jovyan/scratch"
REMOTE_PROJECT_DIR = f"{REMOTE_SCRATCH_DIR}/agenticnav_paper_replica"

# Common HTTP Headers
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Content-Type": "application/json"
}

def get_ssl_context() -> ssl.SSLContext:
    """Returns an unverified SSL context for self-signed cluster certificates."""
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def api_request(url: str, method: str = "GET", payload: dict = None, timeout: int = 30, retries: int = 3) -> dict:
    """Convenience helper for JSON API requests against the cluster dashboard."""
    ctx = get_ssl_context()
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=HEADERS, method=method)
    import time
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
                content = resp.read().decode("utf-8")
                if not content.strip():
                    return {"status": resp.status}
                return json.loads(content)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"[HTTP {e.code} Error Body]: {err_body}")
            raise
        except (TimeoutError, urllib.error.URLError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    return {}

