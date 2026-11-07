import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "1868")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

print("Ep 1868 Instruction:", ep.get("instruction", {}).get("instruction_text", "N/A"))

for v in range(4):
    d = obs.depth_views[v]
    print(f"\nView {v}: min={d.min():.2f}, max={d.max():.2f}, center={d[240, 240]:.2f}")
    # sample depths along horizontal center row y=240
    row_samples = [f"{d[240, x]:.2f}" for x in range(40, 480, 80)]
    print(f"  Horiz profile y=240 (x=40..440): {row_samples}")
    # sample depths along bottom row y=450
    bot_samples = [f"{d[450, x]:.2f}" for x in range(40, 480, 80)]
    print(f"  Bottom profile y=450 (x=40..440): {bot_samples}")

emb.close()

