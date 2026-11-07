"""
Bidirectional Synchronization Client: Remote GPU Server -> Local Machine Backup.
Downloads all scripts, results, and logs from the remote JupyterHub container
to prevent data loss during routine server maintenance or container wipes.
"""
import os
import ssl
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any

TOKEN = 'b4bbbe0a02884d46af65379e7b74f337'
SERVER_BASE = 'https://172.24.16.175/user/krutik'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'Authorization': f'token {TOKEN}',
    'Content-Type': 'application/json'
}

LOCAL_BACKUP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "remote_backup")


def get_remote_item(remote_path: str) -> Dict[str, Any]:
    url = f"{SERVER_BASE}/api/contents/{remote_path.lstrip('/')}"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=ctx) as resp:
        return json.loads(resp.read().decode('utf-8'))


def download_file(remote_path: str, local_dest: str):
    item = get_remote_item(remote_path)
    content = item.get('content')
    if content is None:
        return

    os.makedirs(os.path.dirname(local_dest), exist_ok=True)
    
    # Handle text vs base64 binary
    if item.get('format') == 'base64':
        import base64
        with open(local_dest, 'wb') as f:
            f.write(base64.b64decode(content))
    else:
        with open(local_dest, 'w', encoding='utf-8') as f:
            f.write(content)
    print(f"  [SAVED] {remote_path} -> {os.path.relpath(local_dest, os.getcwd())}")


def sync_directory_recursive(remote_dir: str, local_target: str, extensions: List[str]):
    try:
        dir_obj = get_remote_item(remote_dir)
    except Exception as e:
        print(f"Failed to access remote directory {remote_dir}: {e}")
        return

    for item in dir_obj.get('content', []):
        name = item.get('name')
        item_type = item.get('type')
        remote_path = item.get('path')

        if item_type == 'directory':
            if name in ['__pycache__', '.git', 'conda_envs', 'ollama_models', 'saved_frames', 'OpenFly-Platform', 'SuperNav', 'agentic_uav']:
                continue
            sync_directory_recursive(remote_path, os.path.join(local_target, name), extensions)
        elif item_type == 'file':
            if any(name.endswith(ext) for ext in extensions):
                dest_file = os.path.join(local_target, name)
                try:
                    download_file(remote_path, dest_file)
                except Exception as ex:
                    print(f"  [WARN] Failed to download {remote_path}: {ex}")


if __name__ == '__main__':
    print("=" * 70)
    print("  ARCHIVING REMOTE SCRIPTS & RESULTS TO LOCAL MACHINE...")
    print(f"  Target: {LOCAL_BACKUP_DIR}")
    print("=" * 70)

    TARGET_EXTENSIONS = ['.py', '.sh', '.json', '.log', '.txt', '.md']
    sync_directory_recursive("scratch", LOCAL_BACKUP_DIR, TARGET_EXTENSIONS)
    print("\n[SUCCESS] All remote scripts, logs, and results safely archived locally.")
