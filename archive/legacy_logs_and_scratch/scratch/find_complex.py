import json

with open("agenticnav_paper_replica/data/episode_list_100.json", "r") as f:
    data = json.load(f)["episodes"]

complex_eps = []
for i, ep in enumerate(data):
    # Select episodes with shortest_path_distance >= 11.0m and distinct instructions
    if ep.get("shortest_path_distance", 0.0) >= 11.0:
        complex_eps.append((i, ep))

print(f"Total complex episodes (>=11m): {len(complex_eps)}")
for idx, (orig_i, ep) in enumerate(complex_eps[:15]):
    eid = ep["episode_id"]
    sp = ep["shortest_path_distance"]
    inst = ep["instruction"]
    start = ep.get("start_position")
    goal = ep.get("goal_position")
    print(f"[{idx:02d}] index {orig_i:02d} ({eid}): dist={sp:.2f}m | start={start} -> goal={goal} | inst: {inst}")

