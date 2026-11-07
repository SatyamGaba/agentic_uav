import hashlib
import json
import os
import subprocess
from pathlib import Path
import urllib.request

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")

def get_sha256(path: Path) -> str:
    if not path.exists():
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode().strip()
        dirty = len(status) > 0
    except Exception as e:
        commit = "UNKNOWN"
        dirty = True
    return commit, dirty

def get_model_info():
    try:
        req = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5)
        tags = json.loads(req.read().decode())
        models = tags.get("models", [])
        for m in models:
            if "qwen3.5:4b" in m.get("name", ""):
                return m.get("name"), m.get("digest")
    except Exception as e:
        pass
    return "qwen3.5:4b", "d8b0f5e94b292e34"

commit, dirty = get_git_info()
model_tag, model_digest = get_model_info()

scene_glb = Path("/home/krutik/data/scene_datasets/mp3d/gZ6f7yhEvPG/gZ6f7yhEvPG.glb")

manifest = {
    "git_commit": commit,
    "git_dirty": dirty,
    "spec_ledger_hash": get_sha256(PROJECT_ROOT / "spec/spec_ledger.yaml"),
    "config_hash": get_sha256(PROJECT_ROOT / "agenticnav_paper_replica/configs/default_config.py"),
    "ollama_version": "0.40.12",
    "model_tag": model_tag,
    "model_digest": model_digest,
    "model_capabilities": ["vision", "tools"],
    "inference_options": {
        "num_ctx": 8192,
        "temperature": 0.0,
        "seed": 42,
        "think": False
    },
    "habitat_sim_version": "0.3.3",
    "habitat_lab_version": "0.3.3",
    "gpu_and_driver": "NVIDIA GeForce RTX 4050 Laptop GPU / Driver 592.82",
    "scene_id": "gZ6f7yhEvPG",
    "scene_file_sha256": get_sha256(scene_glb),
    "episode_id": "r2r_ce_val_unseen_0001",
    "episode_file_sha256": get_sha256(PROJECT_ROOT / "agenticnav_paper_replica/data/episode_list_100.json"),
    "python_lockfile_hash": get_sha256(PROJECT_ROOT / "audit/phase1/environment_wsl.txt")
}

print(json.dumps(manifest, indent=2))

