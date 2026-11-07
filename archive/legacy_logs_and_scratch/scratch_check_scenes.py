import json

with open('/home/jovyan/scratch/benchmark/seen.json') as f:
    data = json.load(f)

scenes = {}
city_episodes = []
for i, ep in enumerate(data):
    s = ep.get('scene', ep.get('map', ep.get('scene_id', 'unknown')))
    scenes[s] = scenes.get(s, 0) + 1
    # Check start position
    pos = ep.get('pos', ep.get('trajectory', []))
    if pos:
        start = pos[0]
        # City map rooftop is near (0, 0) or within 100m
        if abs(start[0]) < 150 and abs(start[1]) < 150:
            city_episodes.append(i)

print(f"Total episodes: {len(data)}")
print(f"Scenes: {scenes}")
print(f"Episodes near center (<150m): {len(city_episodes)} -> {city_episodes[:10]}")
if city_episodes:
    first_ep = data[city_episodes[0]]
    print(f"Sample instruction: {first_ep.get('instruction', first_ep.get('gpt_instruction', ''))}")
    print(f"Start pos: {first_ep.get('pos', [])[0] if first_ep.get('pos') else 'none'}")
    print(f"End pos: {first_ep.get('pos', [])[-1] if first_ep.get('pos') else 'none'}")
