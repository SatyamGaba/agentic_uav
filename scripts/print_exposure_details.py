import json
with open("audit/freeze/exposure.json") as f:
    d = json.load(f)
exp = sorted(list(d["exposed_episodes"].keys()), key=int)
unexp = sorted(list(d["unexposed_episode_ids"]), key=int)
print(f"Exposed ({len(exp)}):", exp)
print(f"Unexposed ({len(unexp)}):", unexp)
