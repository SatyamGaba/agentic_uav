import json

with open("agenticnav_paper_replica/data/episode_list_100.json", "r") as f:
    data = json.load(f)

episodes = data["episodes"]
print(f"Total episodes in list: {len(episodes)}")

# Sort or check distances
dists = [ep.get("shortest_path_distance", 0.0) for ep in episodes]
print(f"Distance range: min={min(dists):.2f}m, max={max(dists):.2f}m, avg={sum(dists)/len(dists):.2f}m")

for i, ep in enumerate(episodes[:25]):
    eid = ep.get("episode_id")
    dist = ep.get("shortest_path_distance", 0.0)
    inst = ep.get("instruction", "")
    print(f"[{i:02d}] {eid} | sp_dist={dist:5.2f}m | {inst}")
