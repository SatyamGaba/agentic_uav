#!/usr/bin/env python3
"""Downloads additional Matterport3D scenes for Multi-Scene Gate Verification (G4.1-G4.8)."""

import os
import urllib.request
from pathlib import Path

BASE_DIR = Path("/home/krutik/data/scene_datasets/mp3d")
SCENES = ["pLe4wQe7qrG", "sT4fr6TAbpF"]
HF_BASE = "https://huggingface.co/datasets/bingCS/mp3d_dataset/resolve/main/mp3d"

def download_file(url: str, dest: Path):
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"Already exists: {dest} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
        return
    print(f"Downloading {url} -> {dest}...")
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            print(f"  {total / 1024 / 1024:.1f} MB downloaded", end="\r", flush=True)
    print(f"\nSaved {dest} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")

def main():
    for s in SCENES:
        s_dir = BASE_DIR / s
        glb_url = f"{HF_BASE}/{s}/{s}.glb"
        nav_url = f"{HF_BASE}/{s}/{s}.navmesh"
        
        download_file(glb_url, s_dir / f"{s}.glb")
        download_file(nav_url, s_dir / f"{s}.navmesh")
    print("\nAll additional scenes successfully downloaded!")

if __name__ == "__main__":
    main()

