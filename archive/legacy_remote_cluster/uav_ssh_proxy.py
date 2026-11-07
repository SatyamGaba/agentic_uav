"""
OpenSSH ProxyCommand bridge for UAV Container over JupyterHub WebSocket.
Used in ~/.ssh/config:
    ProxyCommand python "c:/Users/Kruti/Desktop/PROG/UAV/uav_ssh_proxy.py"
"""
import ssl
import sys
import json
import urllib.request
import websocket

SERVER_BASE = "https://172.24.16.175/user/krutik"
WS_BASE = "wss://172.24.16.175/user/krutik"
TOKEN = "b4bbbe0a02884d46af65379e7b74f337"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Content-Type": "application/json"
}

def get_or_create_terminal():
    req = urllib.request.Request(f"{SERVER_BASE}/api/terminals", headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            terms = json.loads(resp.read().decode('utf-8'))
        if terms:
            return terms[0]['name']
    except Exception:
        pass
    req = urllib.request.Request(f"{SERVER_BASE}/api/terminals", headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, context=ctx) as resp:
        term = json.loads(resp.read().decode('utf-8'))
        return term['name']

def main():
    term_id = get_or_create_terminal()
    ws = websocket.create_connection(
        f"{WS_BASE}/terminals/websocket/{term_id}",
        header=[f"Authorization: token {TOKEN}"],
        sslopt={"cert_reqs": ssl.CERT_NONE}
    )
    
    # Forward stdin -> WS, WS -> stdout
    import threading
    def reader():
        while True:
            try:
                msg = ws.recv()
                data = json.loads(msg)
                if data[0] == "stdout":
                    sys.stdout.buffer.write(data[1].encode('utf-8', errors='replace'))
                    sys.stdout.buffer.flush()
            except Exception:
                break
                
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    
    try:
        while True:
            chunk = sys.stdin.buffer.read(1024)
            if not chunk:
                break
            ws.send(json.dumps(["stdin", chunk.decode('utf-8', errors='replace')]))
    except Exception:
        pass
    finally:
        ws.close()

if __name__ == "__main__":
    main()
