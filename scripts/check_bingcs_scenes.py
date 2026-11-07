import urllib.request
import json

val_scenes = [
    "EU6Fwq7SyZv", "JF19kD82Mey", "X7NsGo2t5Ab", "Z6R7ygndpVU",
    "cVre9nv62Rh", "dhjEzFoEJ8e", "gZ6f7yhEvPG", "oLBMNvg9in8",
    "pLe4wQe7qrG", "r1Q1Z4BcVte", "sT4fr6TAbpF"
]

url = 'https://huggingface.co/api/datasets/bingCS/mp3d_dataset'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req, timeout=15) as r:
    data = json.loads(r.read().decode())
    siblings = [s['rfilename'] for s in data.get('siblings', [])]

print(f"Total files in bingCS/mp3d_dataset: {len(siblings)}")
found = {}
for s in val_scenes:
    glb = f"mp3d/{s}/{s}.glb"
    nav = f"mp3d/{s}/{s}.navmesh"
    has_glb = glb in siblings
    has_nav = nav in siblings
    found[s] = {"glb": has_glb, "navmesh": has_nav}
    print(f"  {s}: glb={has_glb}, navmesh={has_nav}")

all_found = all(f["glb"] and f["navmesh"] for f in found.values())
print(f"All 11 scenes present: {all_found}")

