"""
Interactive Terminal Client for Remote UAV JupyterHub Container.
Run this script from Windows terminal:
    python connect_remote.py
"""
import os
import ssl
import sys
import time
import json
import threading
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

def create_fresh_terminal():
    req = urllib.request.Request(f"{SERVER_BASE}/api/terminals", headers=HEADERS, method="POST")
    with urllib.request.urlopen(req, context=ctx) as resp:
        term = json.loads(resp.read().decode('utf-8'))
        return term['name']

def delete_terminal(term_id):
    req = urllib.request.Request(f"{SERVER_BASE}/api/terminals/{term_id}", headers=HEADERS, method="DELETE")
    try:
        with urllib.request.urlopen(req, context=ctx) as resp:
            pass
    except Exception:
        pass

def main():
    print("=" * 65)
    print("  Connecting to Clean Bash Shell on Remote GPU Server...")
    print("=" * 65)
    
    # Always spawn a dedicated, fresh terminal so no simulators occupy stdin
    term_id = create_fresh_terminal()
    ws_url = f"{WS_BASE}/terminals/websocket/{term_id}"
    
    ws = websocket.create_connection(
        ws_url,
        header=[f"Authorization: token {TOKEN}"],
        sslopt={"cert_reqs": ssl.CERT_NONE}
    )
    
    print(f"Connected! (Terminal Session #{term_id})")
    print("Type bash commands below. Type 'exit' to disconnect.\n")
    
    stop_event = threading.Event()
    
    def reader():
        while not stop_event.is_set():
            try:
                ws.settimeout(0.2)
                msg = ws.recv()
                data = json.loads(msg)
                if data[0] == "stdout":
                    sys.stdout.buffer.write(data[1].encode('utf-8', errors='replace'))
                    sys.stdout.buffer.flush()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception:
                break
                
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    
    # Initialize shell prompt
    time.sleep(0.3)
    ws.send(json.dumps(["stdin", "\n"]))
    
    try:
        while not stop_event.is_set():
            try:
                cmd = input()
            except EOFError:
                break
            if cmd.strip().lower() == "exit":
                break
            ws.send(json.dumps(["stdin", f"{cmd}\n"]))
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ws.close()
        delete_terminal(term_id)
        print("\nDisconnected from remote terminal.")

if __name__ == "__main__":
    main()
