#!/usr/bin/env python3
"""Build a deterministic R2R-CE val_unseen subset from official annotations.

The source file is the official ``val_unseen.json.gz`` from
``R2R_VLNCE_v1-3``. Episodes are restricted to scenes that exist locally and
are sampled deterministically with a fixed seed.
"""

import argparse
import gzip
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(
    "/home/krutik/data/datasets/R2R_VLNCE_v1-3/R2R_VLNCE_v1-3/"
    "val_unseen/val_unseen.json.gz"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "agenticnav_paper_replica/data/r2rce_val_unseen_17.json"

AVAILABLE_SCENE_IDS = (
    "EU6Fwq7SyZv",
    "JF19kD82Mey",
    "gZ6f7yhEvPG",
    "oLBMNvg9in8",
    "pLe4wQe7qrG",
    "sT4fr6TAbpF",
)

# Stratified allocation for 17 episodes across the three val_unseen scenes
# that are present locally. Order is fixed for reproducibility.
SCENE_QUOTAS = {
    "EU6Fwq7SyZv": 7,
    "oLBMNvg9in8": 7,
    "pLe4wQe7qrG": 3,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_episodes(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError(f"Unexpected R2R-CE data format: {path}")


def _scene_id_from_path(scene_path: str) -> str:
    return Path(scene_path).stem


def _normalize(episode: Dict[str, Any]) -> Dict[str, Any]:
    instruction = episode.get("instruction", "")
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "")
    goals = episode.get("goals") or []
    if not goals:
        raise ValueError(f"Episode {episode.get('episode_id')} has no goal.")
    goal = goals[0]
    return {
        "episode_id": f"r2r_ce_val_unseen_{int(episode['episode_id']):04d}",
        "original_episode_id": int(episode["episode_id"]),
        "trajectory_id": episode.get("trajectory_id"),
        "scene_id": _scene_id_from_path(str(episode["scene_id"])),
        "source_scene_id": str(episode["scene_id"]),
        "instruction": str(instruction).strip(),
        "start_position": [float(x) for x in episode["start_position"]],
        "start_rotation": [float(x) for x in episode["start_rotation"]],
        "goal_position": [float(x) for x in goal["position"]],
        "goal_radius": float(goal.get("radius", 3.0)),
        "shortest_path_distance": float(
            episode.get("info", {}).get("geodesic_distance", 0.0)
        ),
        "reference_path": [
            [float(x) for x in point] for point in episode.get("reference_path", [])
        ],
    }


def _stratified_sample(
    episodes: Sequence[Dict[str, Any]],
    quotas: Dict[str, int],
    seed: int,
) -> List[Dict[str, Any]]:
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for episode in episodes:
        scene = _scene_id_from_path(str(episode["scene_id"]))
        by_scene.setdefault(scene, []).append(episode)

    rng = random.Random(seed)
    selected: List[Dict[str, Any]] = []
    for scene in sorted(quotas):
        pool = sorted(by_scene.get(scene, []), key=lambda e: int(e["episode_id"]))
        quota = quotas[scene]
        if len(pool) < quota:
            raise ValueError(
                f"Scene {scene} has {len(pool)} episodes, cannot sample {quota}."
            )
        selected.extend(rng.sample(pool, quota))

    selected.sort(key=lambda e: int(e["episode_id"]))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Build official R2R-CE val_unseen subset")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(
            f"Official R2R-CE annotations not found: {args.source}. "
            "Download R2R_VLNCE_v1-3.zip and extract it first."
        )

    source_episodes = _load_episodes(args.source)
    usable = [
        episode
        for episode in source_episodes
        if _scene_id_from_path(str(episode["scene_id"])) in SCENE_QUOTAS
    ]
    selected = _stratified_sample(usable, SCENE_QUOTAS, args.seed)
    normalized = [_normalize(episode) for episode in selected]

    payload = {
        "metadata": {
            "dataset": "R2R-CE",
            "version": "R2R_VLNCE_v1-3",
            "split": "val_unseen",
            "source_file": str(args.source),
            "source_sha256": _sha256(args.source),
            "selection": "stratified random sample without replacement",
            "selection_seed": args.seed,
            "scene_quotas": SCENE_QUOTAS,
            "available_scene_ids": list(AVAILABLE_SCENE_IDS),
            "note": (
                "Subset of the official 1,839-episode R2R-CE val_unseen split. "
                "Only episodes whose Matterport3D scene is locally available are used."
            ),
        },
        "episodes": normalized,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(normalized)} episodes to {args.output}")
    for scene in sorted(SCENE_QUOTAS):
        ids = [e["episode_id"] for e in normalized if e["scene_id"] == scene]
        print(f"  {scene}: {len(ids)} episodes -> {ids}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
