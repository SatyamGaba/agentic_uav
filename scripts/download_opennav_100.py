import gzip
import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

url = "https://raw.githubusercontent.com/YanyuanQiao/Open-Nav/main/data/datasets/R2R_VLNCE_v1-2_preprocessed/val_unseen/OpenNav_R2R-CE_100_bertidx.json.gz"
dest = Path("agenticnav_paper_replica/data/OpenNav_R2R-CE_100_bertidx.json.gz")

print(f"Downloading from {url} to {dest}...")
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
    f.write(resp.read())

sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
print(f"Downloaded {dest.stat().st_size} bytes. SHA-256: {sha256}")

with gzip.open(dest, "rt", encoding="utf-8") as f:
    data = json.load(f)

eps = data.get("episodes", data if isinstance(data, list) else [])
print(f"Total episodes in OpenNav_R2R-CE_100: {len(eps)}")

def clean_scene(sid):
    parts = sid.split("/")
    if len(parts) >= 2:
        return parts[-2] if parts[-1].endswith(".glb") else parts[1]
    return Path(sid).stem

scenes = Counter(clean_scene(e["scene_id"]) for e in eps)
print(f"\nScene distribution across {len(scenes)} scenes:")
for s, c in sorted(scenes.items()):
    print(f"  {s}: {c} episodes")

ep_ids = [e["episode_id"] for e in eps]
print(f"\nEpisode IDs (first 10): {ep_ids[:10]}")
print(f"Episode IDs (last 5): {ep_ids[-5:]}")

