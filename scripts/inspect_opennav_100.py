import gzip
import json
from collections import Counter
import hashlib

gz_path = "agenticnav_paper_replica/data/OpenNav_R2R-CE_100_bertidx.json.gz"
with open(gz_path, "rb") as f:
    gz_sha256 = hashlib.sha256(f.read()).hexdigest()

print(f"GZ File SHA256: {gz_sha256}")

with gzip.open(gz_path, "rt", encoding="utf-8") as f:
    data = json.load(f)

episodes = data.get("episodes", data) if isinstance(data, dict) else data
print(f"Total episodes: {len(episodes)}")

scenes = Counter(ep["scene_id"].split("/")[-1].split(".")[0] for ep in episodes)
print(f"Scene counts ({len(scenes)} scenes):")
for s, c in sorted(scenes.items()):
    print(f"  {s}: {c}")

ep_ids = [ep["episode_id"] for ep in episodes]
print("First 10 ep IDs:", ep_ids[:10])
print("Last 10 ep IDs:", ep_ids[-10:])

# Sample episode fields
sample = episodes[0]
print("\nSample episode keys:", list(sample.keys()))
print("Sample episode ID:", sample.get("episode_id"))
print("Sample scene_id:", sample.get("scene_id"))
print("Sample trajectory length:", len(sample.get("reference_path", [])))
print("Sample instruction:", sample.get("instruction", {}).get("instruction_text"))

# Format standard opennav_r2rce_100.json
out_path = "agenticnav_paper_replica/data/opennav_r2rce_100.json"
clean_episodes = []
for ep in episodes:
    # Standardize scene_id to relative/clean path if needed
    scene_basename = ep["scene_id"].split("/")[-1].split(".")[0]
    clean_ep = dict(ep)
    # Ensure standard scene_id points to mp3d scene format
    clean_ep["scene_basename"] = scene_basename
    clean_episodes.append(clean_ep)

payload = {
    "dataset": "OpenNav_R2R-CE_100",
    "source_url": "https://raw.githubusercontent.com/YanyuanQiao/Open-Nav/main/data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz",
    "source_gz_sha256": gz_sha256,
    "episodes": clean_episodes
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"\nWrote standardized dataset to {out_path} with {len(clean_episodes)} episodes.")

