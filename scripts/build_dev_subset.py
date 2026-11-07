import gzip
import json
import random

source_path = "/home/krutik/data/datasets/R2R_VLNCE_v1-3/R2R_VLNCE_v1-3/train/train.json.gz"
out_path = "agenticnav_paper_replica/data/r2rce_dev_20.json"

with gzip.open(source_path, "rt", encoding="utf-8") as f:
    data = json.load(f)

episodes = data.get("episodes", data)
dev_scenes = ["gZ6f7yhEvPG", "sT4fr6TAbpF", "JF19kD82Mey"]

# Group by scene
by_scene = {s: [] for s in dev_scenes}
for ep in episodes:
    scene_base = ep["scene_id"].split("/")[-1].split(".")[0]
    if scene_base in by_scene:
        by_scene[scene_base].append(ep)

# Deterministic sample: 3 from gZ6, 10 from sT4, 7 from JF19
quotas = {"gZ6f7yhEvPG": 3, "sT4fr6TAbpF": 10, "JF19kD82Mey": 7}
rng = random.Random(42)

selected = []
for s, q in quotas.items():
    pool = by_scene[s]
    rng.shuffle(pool)
    for ep in pool[:q]:
        clean_ep = dict(ep)
        clean_ep["scene_basename"] = s
        selected.append(clean_ep)

payload = {
    "dataset": "r2rce_dev_20_non_test",
    "description": "20 deterministic dev episodes from non-test scenes (train split) for tuning and deadlock auditing",
    "episodes": selected
}

with open(out_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)

print(f"Created {out_path} with {len(selected)} episodes across scenes: {quotas}")

