import ssl
import json
import urllib.request
import sys
import os

TOKEN = 'b4bbbe0a02884d46af65379e7b74f337'
SERVER_BASE = 'https://172.24.16.175/user/krutik'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def upload_file(local_path: str, remote_path: str):
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
        print(f"Uploaded {local_path} -> {remote_path} (Status: {resp.status})")

if __name__ == '__main__':
    if len(sys.argv) >= 3:
        upload_file(sys.argv[1], sys.argv[2])
    else:
        print("Usage: upload_to_remote.py <local> <remote>")
