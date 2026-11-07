import json

with open("agenticnav_paper_replica/data/episode_list_100.json") as f:
    data = json.load(f)

print(f"Total episodes: {len(data['episodes'])}")
scenes = set()
for ep in data['episodes']:
    scenes.add(ep.get('scene_id'))

print(f"Scenes referenced ({len(scenes)}): {sorted(list(scenes))}")
for ep in data['episodes'][:5]:
    print(ep['episode_id'], ep['scene_id'], ep['start_position'], ep.get('goal_position', ep.get('goals')))

