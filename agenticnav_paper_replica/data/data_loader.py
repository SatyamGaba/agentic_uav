"""Episode loader for official R2R-CE data plus a synthetic CI-only generator.

``episode_list_100.json`` is deprecated and is NOT an R2R-CE split. Benchmarks must
use ``r2rce_val_unseen_17.json`` or the original official annotations.
"""

import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_episodes(
    path: str,
    max_episodes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Loads episodes from a JSON or JSON.GZ file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Episode file not found: {path}")

    if p.suffix == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

    if isinstance(data, dict) and "episodes" in data:
        episodes = data["episodes"]
    elif isinstance(data, list):
        episodes = data
    else:
        raise ValueError(f"Unexpected data format in {path}")

    if max_episodes is not None:
        episodes = episodes[:max_episodes]

    return episodes


def generate_synthetic_dev_episodes(num_episodes: int = 5) -> List[Dict[str, Any]]:
    """Generates synthetic episodes for CI only. Never use these for benchmarks."""
    episodes = []
    instructions = [
        "Walk straight out of the bedroom into the hallway and turn left towards the stairs.",
        "Proceed down the corridor, pass the open doorway on your right, and stop in front of the kitchen island.",
        "Exit the bathroom, turn right into the living room, and stop next to the large sofa.",
        "Go straight ahead past the dining table, turn left down the hall, and wait by the front door.",
        "Walk forward into the office, turn left towards the bookshelf, and stop.",
    ]

    for i in range(num_episodes):
        instr = instructions[i % len(instructions)]
        episodes.append({
            "episode_id": f"synthetic_{i+1:03d}",
            "scene_id": f"scene_synth_{i%2}",
            "instruction": instr,
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": [0.0, 0.0, 0.0, 1.0],
            "goal_position": [0.0, 0.0, -8.0 - 2.0 * (i % 3)],
            "shortest_path_distance": 8.0 + 2.0 * (i % 3),
        })

    return episodes

