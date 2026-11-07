"""
Interactive Terminal Client for Remote UAV JupyterHub Container.
Bridges your Windows terminal directly into a clean bash shell on the remote server.

Usage:
    python remote/connect_terminal.py
"""
import sys
import os
import ssl
import json
import time
import threading
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import JUPYTER_BASE, WS_BASE, TOKEN, HEADERS, get_ssl_context

def create_fresh_terminal() -> str:
    """Spawns a clean, dedicated terminal session in the user container."""
    url = f"{JUPYTER_BASE}/api/terminals"
    req = urllib.request.Request(url, headers=HEADERS, method="POST")
    ctx = get_ssl_context()
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        term = json.loads(resp.read().decode("utf-8"))
        return term["name"]

def delete_terminal(term_id: str):
    """Cleans up the terminal session on disconnect."""
    url = f"{JUPYTER_BASE}/api/terminals/{term_id}"
    req = urllib.request.Request(url, headers=HEADERS, method="DELETE")
    ctx = get_ssl_context()
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=5):
            pass
    except Exception:
        pass

def main():
    try:
        import websocket
    except ImportError:
        print("[ERROR] 'websocket-client' package is required.")
        print("Install it with: pip install websocket-client")
        sys.exit(1)

    print("=" * 65)
    print("      Connecting to Remote Shell on GPU Server...            ")
    print("=" * 65)

    try:
        term_id = create_fresh_terminal()
    except urllib.error.HTTPError as e:
        if e.code == 503:
            print("\n[NOTICE] User container is currently STOPPED (GPU 0 is occupied by another session).")
            print("Good news: You can still submit and run all GPU jobs directly via:")
            print("    python remote/submit_job.py --name 'my-job' --cmd '...' --time 30")
            print("Batch jobs run on dedicated GPU 1 and do NOT require an open terminal!")
        else:
            print(f"[ERROR] Could not allocate terminal (HTTP {e.code}): {e.reason}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        print("Tip: Ensure campus VPN is connected and reachable.")
        sys.exit(1)

    ws_url = f"{WS_BASE}/terminals/websocket/{term_id}"
    try:
        ws = websocket.create_connection(
            ws_url,
            header=[f"Authorization: token {TOKEN}"],
            sslopt={"cert_reqs": ssl.CERT_NONE}
        )
    except Exception as e:
        print(f"[ERROR] WebSocket connection failed: {e}")
        delete_terminal(term_id)
        sys.exit(1)

    print(f"[CONNECTED] Clean Terminal Session #{term_id}")
    print("Type bash commands below. Type 'exit' to disconnect.\n")

    stop_event = threading.Event()

    def receive_output():
        while not stop_event.is_set():
            try:
                msg = ws.recv()
                if not msg:
                    break
                data = json.loads(msg)
                if data[0] == "stdout":
                    sys.stdout.write(data[1])
                    sys.stdout.flush()
            except Exception:
                break

    recv_thread = threading.Thread(target=receive_output, daemon=True)
    recv_thread.start()

    try:
        while not stop_event.is_set():
            cmd = input()
            if cmd.strip().lower() in ["exit", "quit"]:
                break
            ws.send(json.dumps(["stdin", cmd + "\n"]))
            time.sleep(0.05)
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        stop_event.set()
        try:
            ws.close()
        except Exception:
            pass
        delete_terminal(term_id)
        print("\n[DISCONNECTED] Terminal session closed cleanly.")

if __name__ == "__main__":
    main()

