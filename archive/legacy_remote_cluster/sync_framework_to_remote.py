import os
import ssl
import json
import urllib.request

TOKEN = 'b4bbbe0a02884d46af65379e7b74f337'
SERVER_BASE = 'https://172.24.16.175/user/krutik'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def upload(local_path, remote_path):
    with open(local_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    url = f"{SERVER_BASE}/api/contents/{remote_path.lstrip('/')}"
    data = json.dumps({'content': content, 'type': 'file', 'format': 'text'}).encode('utf-8')
    req = urllib.request.Request(
        url,
        data=data,
        headers={'Authorization': f'token {TOKEN}', 'Content-Type': 'application/json'},
        method='PUT'
    )
    with urllib.request.urlopen(req, context=ctx) as resp:
        print(f"Synced {remote_path} ({resp.status})")

def sync_dir(local_dir, remote_prefix):
    for root, dirs, files in os.walk(local_dir):
        if '__pycache__' in root:
            continue
        for file in files:
            if not file.endswith('.py'):
                continue
            local_full = os.path.join(root, file)
            rel = os.path.relpath(local_full, local_dir).replace('\\', '/')
            rem = f"{remote_prefix}/{rel}"
            upload(local_full, rem)

if __name__ == '__main__':
    print("Syncing framework to remote...")
    sync_dir("framework", "scratch/framework")
    upload("run_framework_benchmark.py", "scratch/run_framework_benchmark.py")
    upload("launch_supernav_airsim.sh", "scratch/launch_supernav_airsim.sh")
    print("Sync complete!")
