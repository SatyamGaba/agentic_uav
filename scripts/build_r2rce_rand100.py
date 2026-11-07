#!/usr/bin/env python3
"""Build the official 100-episode R2R-CE benchmark dataset ('rand100').

Extracts indices 0-99 (the exact standard 100 episodes evaluated in AgenticNav,
SmartWay, and Open-Nav) from official R2R-CE val_unseen.json.gz.
Preserves strict provenance, source hashes, and reference trajectories.
"""

import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = Path("/home/krutik/data/datasets/R2R_VLNCE_v1-3/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz")
OUTPUT_PATH = PROJECT_ROOT / "agenticnav_paper_replica/data/r2rce_val_unseen_100_rand.json"

EXPECTED_SOURCE_SHA256 = "d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8"

def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def clean_scene_id(sid: str) -> str:
    parts = sid.split("/")
    if len(parts) >= 2:
        return parts[1]
    return Path(sid).stem

def normalize_episode(ep: dict, idx: int) -> dict:
    instruction = ep.get("instruction", "")
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "")
    goals = ep.get("goals") or []
    if not goals:
        raise ValueError(f"Episode {ep.get('episode_id')} at index {idx} has no goal.")
    goal = goals[0]

    return {
        "index": idx,
        "episode_id": f"r2r_ce_val_unseen_{int(ep['episode_id']):04d}",
        "original_episode_id": int(ep["episode_id"]),
        "trajectory_id": ep.get("trajectory_id"),
        "scene_id": clean_scene_id(str(ep["scene_id"])),
        "source_scene_id": str(ep["scene_id"]),
        "instruction": str(instruction).strip(),
        "start_position": [float(x) for x in ep["start_position"]],
        "start_rotation": [float(x) for x in ep["start_rotation"]],
        "goal_position": [float(x) for x in goal["position"]],
        "goal_radius": float(goal.get("radius", 3.0)),
        "shortest_path_distance": float(ep.get("info", {}).get("geodesic_distance", 0.0)),
        "reference_path": [[float(x) for x in pt] for pt in ep.get("reference_path", [])],
    }

def main():
    if not SOURCE_PATH.exists():
        print(f"Error: Source file {SOURCE_PATH} not found.")
        sys.exit(1)

    print(f"Verifying source SHA-256 for {SOURCE_PATH}...")
    source_sha = compute_sha256(SOURCE_PATH)
    print(f"Source SHA-256: {source_sha}")
    if source_sha != EXPECTED_SOURCE_SHA256:
        print(f"WARNING: SHA-256 does not match expected {EXPECTED_SOURCE_SHA256}")

    print("Loading episodes from val_unseen.json.gz...")
    with gzip.open(SOURCE_PATH, "rt", encoding="utf-8") as f:
        data = json.load(f)

    all_episodes = data["episodes"]
    print(f"Total episodes in val_unseen: {len(all_episodes)}")

    # Extract first 100 episodes (indices 0..99)
    rand100_raw = all_episodes[:100]
    normalized_100 = [normalize_episode(ep, i) for i, ep in enumerate(rand100_raw)]

    scene_counts = Counter(e["scene_id"] for e in normalized_100)
    print("\nScene distribution in rand100 (indices 0..99):")
    for scene, cnt in sorted(scene_counts.items()):
        print(f"  {scene}: {cnt} episodes")

    payload = {
        "metadata": {
            "dataset": "R2R-CE",
            "split": "val_unseen",
            "name": "r2rce_val_unseen_100_rand",
            "description": "First 100 episodes (indices 0-99) of official R2R-CE val_unseen, standard benchmark for AgenticNav / Open-Nav / SmartWay",
            "source_file": str(SOURCE_PATH),
            "source_sha256": source_sha,
            "num_episodes": len(normalized_100),
            "scene_distribution": dict(scene_counts),
        },
        "episodes": normalized_100,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    out_sha = compute_sha256(OUTPUT_PATH)
    print(f"\nWrote 100 episodes to {OUTPUT_PATH}")
    print(f"Output SHA-256: {out_sha}")

if __name__ == "__main__":
    main()

