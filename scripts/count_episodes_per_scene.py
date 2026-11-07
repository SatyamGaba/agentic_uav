import json
from collections import Counter

with open("agenticnav_paper_replica/data/episode_list_100.json") as f:
    data = json.load(f)

counts = Counter(ep.get("scene_id") for ep in data["episodes"])
print("Episodes per scene:")
for scene, count in sorted(counts.items()):
    print(f"  {scene}: {count}")

