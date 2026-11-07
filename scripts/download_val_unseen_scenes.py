#!/usr/bin/env python3
"""Downloads missing Matterport3D val_unseen scenes from bingCS/mp3d_dataset."""

import os
import sys
import time
import urllib.request
from pathlib import Path

BASE_DIR = Path("/home/krutik/data/scene_datasets/mp3d")
HF_BASE = "https://huggingface.co/datasets/bingCS/mp3d_dataset/resolve/main/mp3d"

# 6 rand100 missing scenes first, then remaining 2
SCENES_TO_DOWNLOAD = [
    "zsNo4HB9uLZ",
    "2azQ1b91cZZ",
    "QUCTc6BB5sX",
    "TbHJrupSAjP",
    "x8F5xyUWy9e",
    "X7HyMhZNoso",
    "8194nk5LbLH",
    "Z6MFQCViBuw"
]

def download_file(url: str, dest: Path, max_retries: int = 5):
    if dest.exists() and dest.stat().st_size > 10000:
        print(f"[SKIP] {dest.name} already exists ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DOWNLOADING] {url} -> {dest.name} (attempt {attempt}/{max_retries})...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(temp_dest, "wb") as f:
                total = 0
                last_print = time.time()
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    total += len(chunk)
                    if time.time() - last_print > 5.0:
                        print(f"  {total / 1024 / 1024:.1f} MB downloaded...")
                        last_print = time.time()
            temp_dest.rename(dest)
            print(f"[SAVED] {dest} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
            return True
        except Exception as e:
            print(f"[ERROR] Attempt {attempt} failed: {e}")
            if temp_dest.exists():
                temp_dest.unlink()
            time.sleep(3 * attempt)
    return False

def main():
    print(f"Starting download of {len(SCENES_TO_DOWNLOAD)} scenes to {BASE_DIR}...")
    success = {}
    for s in SCENES_TO_DOWNLOAD:
        s_dir = BASE_DIR / s
        glb_url = f"{HF_BASE}/{s}/{s}.glb"
        nav_url = f"{HF_BASE}/{s}/{s}.navmesh"
        
        ok_nav = download_file(nav_url, s_dir / f"{s}.navmesh")
        ok_glb = download_file(glb_url, s_dir / f"{s}.glb")
        success[s] = ok_nav and ok_glb

    print("\nDownload Summary:")
    all_ok = True
    for s, ok in success.items():
        print(f"  {s}: {'SUCCESS' if ok else 'FAILED'}")
        if not ok:
            all_ok = False
            
    if all_ok:
        print("\nAll requested scenes downloaded successfully!")
    else:
        print("\nSome scenes failed to download.")
        sys.exit(1)

if __name__ == "__main__":
    main()

