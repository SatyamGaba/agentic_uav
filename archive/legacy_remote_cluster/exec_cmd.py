import ssl
import json
import time
import websocket
import urllib.request
import sys

TOKEN = 'b4bbbe0a02884d46af65379e7b74f337'
SERVER_BASE = 'https://172.24.16.175/user/krutik'
WS_BASE = 'wss://172.24.16.175/user/krutik'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def run_remote(command: str, wait_sec: float = 6.0):
    # Create fresh terminal with resilient retry
    req = urllib.request.Request(f'{SERVER_BASE}/api/terminals', headers={'Authorization': f'token {TOKEN}'}, method='POST')
    term_id = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=25.0) as resp:
                term_id = json.loads(resp.read().decode('utf-8'))['name']
                break
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2.0)

    ws = websocket.create_connection(
        f'{WS_BASE}/terminals/websocket/{term_id}',
        header=[f'Authorization: token {TOKEN}'],
        sslopt={'cert_reqs': ssl.CERT_NONE}
    )

    ws.send(json.dumps(['stdin', f"{command}\n"]))

    out = []
    end = time.time() + wait_sec
    ws.settimeout(0.5)
    while time.time() < end:
        try:
            m = ws.recv()
            d = json.loads(m)
            if d[0] == 'stdout':
                out.append(d[1])
        except Exception:
            pass
    ws.close()

    # Delete terminal
    try:
        del_req = urllib.request.Request(f'{SERVER_BASE}/api/terminals/{term_id}', headers={'Authorization': f'token {TOKEN}'}, method='DELETE')
        urllib.request.urlopen(del_req, context=ctx)
    except Exception:
        pass

    return "".join(out)

if __name__ == '__main__':
    import sys
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timeout', type=float, default=45.0)
    parser.add_argument('cmd', nargs='*')
    args = parser.parse_args()
    command = " ".join(args.cmd) if args.cmd else "whoami"
    print(run_remote(command, args.timeout))
