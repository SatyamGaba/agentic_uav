import json

dev = json.load(open("agenticnav_paper_replica/data/r2rce_dev_20.json"))
val100 = json.load(open("agenticnav_paper_replica/data/opennav_r2rce_100.json"))

dev_eps = dev["episodes"]
val_eps = val100["episodes"]

print(f"Dev episodes count: {len(dev_eps)}")
print(f"Val 100 episodes count: {len(val_eps)}")

dev_scenes = set(e["scene_id"] for e in dev_eps)
val_scenes = set(e["scene_id"] for e in val_eps)

print("Dev scenes:", dev_scenes)
print("Val scenes:", val_scenes)
print("Scene intersection:", dev_scenes.intersection(val_scenes))

# Check ID collision vs actual trajectory overlap
id_intersection = set(str(e["episode_id"]) for e in dev_eps).intersection(set(str(e["episode_id"]) for e in val_eps))
print("String episode_id intersection between dev_20 and val100:", id_intersection)

for epid in id_intersection:
    d = [e for e in dev_eps if str(e["episode_id"]) == epid][0]
    v = [e for e in val_eps if str(e["episode_id"]) == epid][0]
    print(f"\nCollision on ID {epid}:")
    print(f"  Dev: scene={d['scene_id']}, traj_id={d.get('trajectory_id')}, start={d['start_position']}")
    print(f"  Val100: scene={v['scene_id']}, traj_id={v.get('trajectory_id')}, start={v['start_position']}")

