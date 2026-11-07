import os
import glob
import json
from pathlib import Path

# Paths
REPO_ROOT = Path(".").resolve()
LOCKED_100_PATH = REPO_ROOT / "agenticnav_paper_replica/data/opennav_r2rce_100.json"
VAL_UNSEEN_FULL_PATH = Path("/home/krutik/data/datasets/R2R_VLNCE_v1-3/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz")

with open(LOCKED_100_PATH, "r", encoding="utf-8") as f:
    locked_data = json.load(f)

locked_episodes = locked_data.get("episodes", locked_data)
locked_ids = {str(ep["episode_id"]): ep for ep in locked_episodes}
print(f"Total locked episodes: {len(locked_ids)}")

# Map of val_unseen index / name to episode_id if applicable
# Load full val_unseen to get exact mapping if needed
import gzip
val_unseen_episodes = []
if VAL_UNSEEN_FULL_PATH.exists():
    with gzip.open(VAL_UNSEEN_FULL_PATH, "rt", encoding="utf-8") as f:
        val_unseen_data = json.load(f)
    val_unseen_episodes = val_unseen_data.get("episodes", val_unseen_data)
    print(f"Loaded {len(val_unseen_episodes)} episodes from official val_unseen.json.gz")

# Index mapping: index in val_unseen -> episode_id
idx_to_epid = {i: str(ep["episode_id"]) for i, ep in enumerate(val_unseen_episodes)}
epid_to_scene = {str(ep["episode_id"]): ep["scene_id"].split("/")[-1].split(".")[0] for ep in val_unseen_episodes}

# Collect all episode IDs and names seen in all historical runs, audits, and scratch files
exposed_records = []

# 1. Check all audit folders
audit_files = list(REPO_ROOT.glob("audit/**/*.json")) + list(REPO_ROOT.glob("audit/**/*.jsonl"))
trace_files = list(REPO_ROOT.glob("runs/**/*.json*")) + list(REPO_ROOT.glob("archive/**/*.json*"))

# Also include hardcoded known past test episodes: e.g. 0764, 0414, 17-ep set
known_probed = ["0764", "764", "0414", "414", "ep_000", "0", "1", "2"]

def register_exposure(source_file, ep_id, scene=None, notes=""):
    ep_str = str(ep_id).strip()
    # Check if ep_str is an index like 0764 or episode_id
    real_id = ep_str
    if ep_str.isdigit():
        val_int = int(ep_str)
        # Could be episode_id directly or index
        if ep_str in locked_ids:
            real_id = ep_str
        elif val_int in idx_to_epid and idx_to_epid[val_int] in locked_ids:
            real_id = idx_to_epid[val_int]
            notes += f" (mapped from index {val_int})"

    in_locked = real_id in locked_ids
    exposed_records.append({
        "source": str(source_file),
        "raw_id": ep_str,
        "resolved_episode_id": real_id,
        "scene": scene or epid_to_scene.get(real_id, "unknown"),
        "in_locked_100": in_locked,
        "notes": notes
    })

# Check 17-episode set
ep17_file = REPO_ROOT / "agenticnav_paper_replica/data/r2rce_val_unseen_17.json"
if ep17_file.exists():
    with open(ep17_file, "r", encoding="utf-8") as f:
        ep17_data = json.load(f)
    for ep in ep17_data.get("episodes", ep17_data):
        register_exposure("r2rce_val_unseen_17.json", ep.get("episode_id"), ep.get("scene_id"), "Stratified 17-ep evaluation subset")

# Check all json files in audit
for fpath in audit_files + trace_files:
    if "opennav_100_eval" in str(fpath) or "baseline_stop" in str(fpath):
        continue # That is the locked baseline itself, not tuning exposure
    try:
        if fpath.suffix == ".json":
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = json.load(f)
            if isinstance(content, dict):
                if "episode_id" in content:
                    register_exposure(fpath, content["episode_id"], content.get("scene_id"), "Audit episode json")
                if "episodes" in content and isinstance(content["episodes"], list):
                    for item in content["episodes"]:
                        if isinstance(item, dict) and "episode_id" in item:
                            register_exposure(fpath, item["episode_id"], item.get("scene_id"), "Audit summary list")
        elif fpath.suffix == ".jsonl":
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if "episode_id" in item:
                            register_exposure(fpath, item["episode_id"], item.get("scene_id"), "Trace line")
    except Exception:
        pass

# Add explicit diagnostic probes mentioned in reviews: 0764, 0414
register_exposure("Pilot A / Diagnostic review", "764", "EU6Fwq7SyZv", "Diagnostic episode 0764 investigated in pilot")
register_exposure("Pilot B / Diagnostic review", "414", "EU6Fwq7SyZv", "Diagnostic episode 0414 investigated in pilot")

# Deduplicate by resolved_episode_id
exposed_in_locked = {}
for rec in exposed_records:
    if rec["in_locked_100"]:
        eid = rec["resolved_episode_id"]
        if eid not in exposed_in_locked:
            exposed_in_locked[eid] = {
                "episode_id": eid,
                "scene": locked_ids[eid].get("scene_id", "").split("/")[-1].split(".")[0],
                "sources": set()
            }
        exposed_in_locked[eid]["sources"].add(str(rec["source"]))

# Convert sets to lists
for eid in exposed_in_locked:
    exposed_in_locked[eid]["sources"] = sorted(list(exposed_in_locked[eid]["sources"]))

unexposed_ids = [eid for eid in locked_ids if eid not in exposed_in_locked]

print(f"\nExposure Audit Results:")
print(f"Total locked episodes: {len(locked_ids)}")
print(f"Exposed locked episodes: {len(exposed_in_locked)}")
print(f"Unexposed locked episodes: {len(unexposed_ids)}")

out_dir = REPO_ROOT / "audit/freeze"
out_dir.mkdir(parents=True, exist_ok=True)
out_file = out_dir / "exposure.json"

payload = {
    "audit_name": "T-A1 Exposure Audit",
    "locked_100_source": "OpenNav_R2R-CE_100_bertidx.json.gz",
    "total_locked_episodes": len(locked_ids),
    "exposed_count": len(exposed_in_locked),
    "unexposed_count": len(unexposed_ids),
    "exposed_episodes": exposed_in_locked,
    "unexposed_episode_ids": unexposed_ids,
    "all_exposure_records": exposed_records
}

with open(out_file, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Wrote exposure audit to {out_file}")
if exposed_in_locked:
    print("\nExposed episodes in locked set:")
    for eid, info in exposed_in_locked.items():
        print(f"  Ep {eid} ({info['scene']}): {len(info['sources'])} sources")

