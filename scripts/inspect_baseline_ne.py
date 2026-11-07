import json
import math

with open("audit/opennav_100_eval/baseline_stop_summary.json") as f:
    d = json.load(f)

eps = d["episodes"]
inf_eps = []
finite_nes = []
for ep in eps:
    ne = ep["metrics"]["navigation_error_m"]
    if math.isinf(ne):
        inf_eps.append((ep["episode_id"], ep["scene_id"]))
    else:
        finite_nes.append(ne)

print(f"Total episodes: {len(eps)}")
print(f"Episodes with inf navigation error: {len(inf_eps)}: {inf_eps}")
if finite_nes:
    print(f"Finite NE count: {len(finite_nes)}, mean: {sum(finite_nes)/len(finite_nes):.2f}m, min: {min(finite_nes):.2f}m, max: {max(finite_nes):.2f}m")

