"""Stage D Freeze Manifest Builder & Integrity Verifier.

Computes cryptographic SHA-256 hashes of all locked parameters, prompts, configs,
geometry algorithms, datasets, and scene manifests.

Any evaluation on the locked 100-episode set must verify these hashes first;
if any file has been modified, execution is aborted immediately.
"""

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FROZEN_FILES = {
    "spec_ledger": "spec/spec_ledger.yaml",
    "config": "agenticnav_paper_replica/configs/default_config.py",
    "prompts": "agenticnav_paper_replica/harness/prompts.py",
    "step_loop": "agenticnav_paper_replica/harness/step_loop.py",
    "runner": "agenticnav_paper_replica/harness/runner.py",
    "executor": "agenticnav_paper_replica/harness/executor.py",
    "safety_region": "agenticnav_paper_replica/geometry/safety_region.py",
    "pinhole": "agenticnav_paper_replica/geometry/pinhole.py",
    "occupancy_grid": "agenticnav_paper_replica/mapping/occupancy_grid.py",
    "bev_renderer": "agenticnav_paper_replica/mapping/bev_renderer.py",
    "locked_100_dataset": "agenticnav_paper_replica/data/opennav_r2rce_100.json",
    "scene_manifest": "audit/interactive_session/mp3d_val_unseen_11_scenes_hashes.json",
    "exposure_audit": "audit/freeze/exposure.json",
    "run_matrix": "audit/freeze/run_matrix.json",
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def build_manifest():
    manifest = {
        "stage": "Stage D Freeze Manifest",
        "total_files_locked": len(FROZEN_FILES),
        "files": {},
    }
    
    for key, rel_path in FROZEN_FILES.items():
        abs_p = PROJECT_ROOT / rel_path
        if not abs_p.exists():
            raise FileNotFoundError(f"Missing required file for freeze: {abs_p}")
        f_hash = sha256_file(abs_p)
        f_size = abs_p.stat().st_size
        manifest["files"][key] = {
            "path": rel_path,
            "sha256": f_hash,
            "size_bytes": f_size,
        }
        
    out_dir = PROJECT_ROOT / "audit/freeze"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "freeze_manifest.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Locked {len(FROZEN_FILES)} files in Stage D freeze manifest: {out_file}")
    for k, v in manifest["files"].items():
        print(f"  {k:<20}: {v['sha256'][:16]}... ({v['path']})")

def verify_freeze() -> bool:
    manifest_file = PROJECT_ROOT / "audit/freeze/freeze_manifest.json"
    if not manifest_file.exists():
        print("Freeze manifest not found!")
        return False
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    mismatches = []
    for key, info in manifest["files"].items():
        p = PROJECT_ROOT / info["path"]
        if not p.exists():
            mismatches.append(f"Missing: {info['path']}")
            continue
        cur_hash = sha256_file(p)
        if cur_hash != info["sha256"]:
            mismatches.append(f"Hash mismatch on {info['path']}: expected {info['sha256']}, got {cur_hash}")
            
    if mismatches:
        print(f"[FAIL] Freeze verification failed with {len(mismatches)} errors:")
        for m in mismatches:
            print(f"  {m}")
        return False
    print("[PASS] Freeze verification: All files match cryptographic SHA-256 hashes exactly.")
    return True

if __name__ == "__main__":
    if "--verify" in sys.argv:
        if not verify_freeze():
            sys.exit(1)
    else:
        build_manifest()

