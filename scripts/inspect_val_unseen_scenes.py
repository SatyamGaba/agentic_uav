import gzip
import json
from collections import Counter

val_unseen_path = '/home/krutik/data/datasets/R2R_VLNCE_v1-3/R2R_VLNCE_v1-3/val_unseen/val_unseen.json.gz'

with gzip.open(val_unseen_path, 'rt') as f:
    data = json.load(f)

eps = data['episodes']
print(f"Total episodes in val_unseen: {len(eps)}")

def clean_scene(sid):
    # e.g. "mp3d/EU6Fwq7SyZv/EU6Fwq7SyZv.glb" -> "EU6Fwq7SyZv"
    parts = sid.split('/')
    if len(parts) >= 2:
        return parts[1]
    return sid

all_scenes = Counter(clean_scene(e['scene_id']) for e in eps)
print(f"\nAll {len(all_scenes)} scenes in val_unseen:")
for scene, cnt in sorted(all_scenes.items()):
    print(f"  {scene}: {cnt} episodes")

rand100 = eps[:100]
scenes_100 = Counter(clean_scene(e['scene_id']) for e in rand100)
print(f"\nScenes in rand100 (indices 0..99):")
for scene, cnt in sorted(scenes_100.items()):
    print(f"  {scene}: {cnt} episodes")

