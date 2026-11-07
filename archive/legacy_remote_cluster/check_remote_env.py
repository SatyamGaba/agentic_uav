import ssl
import json
import time
import websocket
import urllib.request

TOKEN = 'b4bbbe0a02884d46af65379e7b74f337'
SERVER_BASE = 'https://172.24.16.175/user/krutik'
WS_BASE = 'wss://172.24.16.175/user/krutik'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

req = urllib.request.Request(f'{SERVER_BASE}/api/terminals', headers={'Authorization': f'token {TOKEN}'}, method='POST')
with urllib.request.urlopen(req, context=ctx) as resp:
    term_id = json.loads(resp.read().decode('utf-8'))['name']
print('Connected to terminal:', term_id)

ws = websocket.create_connection(f'{WS_BASE}/terminals/websocket/{term_id}', header=[f'Authorization: token {TOKEN}'], sslopt={'cert_reqs': ssl.CERT_NONE})

commands = [
    "whoami",
    "uptime",
    "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader",
    "ps aux | grep -v grep | grep -E 'AirVLN|Xvfb|ollama'",
    "ls -la /home/jovyan/scratch/airsim_unreal_run/"
]

cmd_str = "; echo '===CMD_SEP==='; ".join(commands) + "\n"
ws.send(json.dumps(['stdin', cmd_str]))

out = []
end_time = time.time() + 4.0
ws.settimeout(0.5)
while time.time() < end_time:
    try:
        m = ws.recv()
        d = json.loads(m)
        if d[0] == 'stdout':
            out.append(d[1])
    except Exception:
        pass
ws.close()

# Cleanup terminal
req_del = urllib.request.Request(f'{SERVER_BASE}/api/terminals/{term_id}', headers={'Authorization': f'token {TOKEN}'}, method='DELETE')
try:
    urllib.request.urlopen(req_del, context=ctx)
except Exception:
    pass

print(''.join(out))
