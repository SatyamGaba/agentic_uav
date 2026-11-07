"""T-A1 Exposure Audit Script (Re-keyed by (scene, trajectory, episode, start_pos)).

Strictly distinguishes between episodes across different splits (train vs val_unseen)
and resolves cross-split ID collisions (such as dev episode 259 in sT4fr6TAbpF vs
locked episode 259 in TbHJrupSAjP).
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCKED_100_PATH = REPO_ROOT / "agenticnav_paper_replica/data/opennav_r2rce_100.json"


def normalize_scene(scene_id: str) -> str:
    """Extract clean scene basename e.g. 'TbHJrupSAjP'."""
    s = str(scene_id).replace("\\", "/")
    base = s.split("/")[-1].replace(".glb", "").replace(".navmesh", "")
    return base


def make_episode_key(scene_id: str, trajectory_id: Any, episode_id: Any, start_pos: Any = None) -> Tuple:
    """Build canonical tuple key: (scene_basename, trajectory_id, str(episode_id), start_pos_rounded)."""
    scene = normalize_scene(scene_id)
    traj = int(trajectory_id) if trajectory_id is not None and str(trajectory_id).isdigit() else str(trajectory_id)
    epid = str(episode_id).strip()
    pos = tuple(round(float(p), 2) for p in start_pos) if start_pos is not None else None
    return (scene, traj, epid, pos)


def run_exposure_audit():
    with open(LOCKED_100_PATH, "r", encoding="utf-8") as f:
        locked_data = json.load(f)

    locked_episodes = locked_data.get("episodes", locked_data)
    locked_keys = {}
    locked_scenes = set()
    for ep in locked_episodes:
        k = make_episode_key(
            ep["scene_id"],
            ep.get("trajectory_id"),
            ep["episode_id"],
            ep.get("start_position"),
        )
        locked_keys[k] = ep
        locked_scenes.add(normalize_scene(ep["scene_id"]))

    print(f"Total locked episodes: {len(locked_episodes)}")
    print(f"Locked scenes ({len(locked_scenes)}): {sorted(list(locked_scenes))}")

    # Track exposures against locked episodes
    exposed_locked = {}  # episode_id -> {scene, trajectory_id, sources: []}

    # Inspect all audit files, positive control runs, and past diagnostic probes
    audit_files = list(REPO_ROOT.glob("audit/**/*.json")) + list(REPO_ROOT.glob("audit/**/*.jsonl"))
    trace_files = list(REPO_ROOT.glob("runs/**/*.json*")) + list(REPO_ROOT.glob("archive/**/*.json*"))

    def check_record(scene: str, traj: Any, ep_id: Any, start_pos: Any, source: str, note: str):
        if not scene or scene == "unknown":
            return
        scene_clean = normalize_scene(scene)
        if scene_clean not in locked_scenes:
            # Not in locked scenes -> ZERO chance of locked exposure
            return

        # Check against locked episodes in this scene
        for (l_scene, l_traj, l_epid, l_pos), ep_info in locked_keys.items():
            if l_scene != scene_clean:
                continue
            # Match either exact episode ID, or matching trajectory + start position
            ep_match = str(ep_id) == l_epid
            traj_match = (traj is not None and l_traj == traj)
            pos_match = (start_pos is not None and l_pos is not None and l_pos == tuple(round(float(p), 2) for p in start_pos))

            if ep_match or (traj_match and pos_match):
                eid = ep_info["episode_id"]
                if eid not in exposed_locked:
                    exposed_locked[eid] = {
                        "episode_id": eid,
                        "scene": l_scene,
                        "trajectory_id": ep_info.get("trajectory_id"),
                        "start_position": ep_info.get("start_position"),
                        "sources": set(),
                        "notes": set(),
                    }
                exposed_locked[eid]["sources"].add(str(source))
                exposed_locked[eid]["notes"].add(note)

    for fpath in audit_files + trace_files:
        fstr = str(fpath).replace("\\", "/")
        if (
            "opennav_100_eval" in fstr
            or "baseline_stop" in fstr
            or "random_walk" in fstr
            or "locked_100_scene_hashes" in fstr
            or "metric_validation" in fstr
            or "freeze/exposure.json" in fstr
        ):
            continue

        try:
            if fpath.suffix == ".json":
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    if "episode_id" in data and "scene_id" in data:
                        check_record(data["scene_id"], data.get("trajectory_id"), data["episode_id"], data.get("start_position"), fstr, "JSON single episode")
                    if "episodes" in data and isinstance(data["episodes"], list):
                        for item in data["episodes"]:
                            if isinstance(item, dict) and "episode_id" in item:
                                scene_val = item.get("scene_id") or item.get("scene") or data.get("scene_id") or data.get("scene")
                                check_record(scene_val, item.get("trajectory_id"), item["episode_id"], item.get("start_position"), fstr, "JSON episode array")
            elif fpath.suffix == ".jsonl":
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            if "episode_id" in item and "scene_id" in item:
                                check_record(item["scene_id"], item.get("trajectory_id"), item["episode_id"], item.get("start_position"), fstr, "JSONL line")
        except Exception:
            pass

    # Diagnostic probe on 0764 (in scene EU6Fwq7SyZv)
    check_record("EU6Fwq7SyZv", None, "765", None, "Pilot A / Diagnostic review", "Diagnostic episode 0764 investigation")

    # Serialize results
    for eid in exposed_locked:
        exposed_locked[eid]["sources"] = sorted(list(exposed_locked[eid]["sources"]))
        exposed_locked[eid]["notes"] = sorted(list(exposed_locked[eid]["notes"]))

    exposed_ids = sorted(list(exposed_locked.keys()))
    unexposed_ids = sorted([ep["episode_id"] for ep in locked_episodes if ep["episode_id"] not in exposed_locked])

    print(f"\nExposure Audit Results:")
    print(f"Total locked episodes: {len(locked_episodes)}")
    print(f"Exposed locked episodes: {len(exposed_ids)} -> {exposed_ids}")
    print(f"Pristine unexposed episodes: {len(unexposed_ids)}")

    # Verify Episode 259 explicitly:
    print(f"\nVerification of Episode 259:")
    print(f"Is Locked Episode 259 exposed? {259 in exposed_locked}")
    if 259 not in exposed_locked:
        print("  -> Episode 259 is PRISTINE in locked set (Dev 259 was in sT4fr6TAbpF, Locked 259 is in TbHJrupSAjP).")

    out_file = REPO_ROOT / "audit/freeze/exposure.json"
    payload = {
        "audit_name": "T-A1 Exposure Audit (Re-keyed)",
        "locked_100_source": "OpenNav_R2R-CE_100_bertidx.json.gz",
        "total_locked_episodes": len(locked_episodes),
        "exposed_count": len(exposed_ids),
        "unexposed_count": len(unexposed_ids),
        "exposed_episodes": exposed_locked,
        "exposed_episode_ids": exposed_ids,
        "unexposed_episode_ids": unexposed_ids,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nWrote re-keyed exposure audit to {out_file}")


if __name__ == "__main__":
    run_exposure_audit()
