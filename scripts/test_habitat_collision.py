import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
# Step 1
obs = emb.step_discrete(2, 8)
# Step 2
obs = emb.step_discrete(-4, 7)
# Step 3
obs = emb.step_discrete(-3, 9)
print("Before forward step, pos:", emb.sim.get_agent(0).get_state().position)
print("Before forward step, rot:", emb.sim.get_agent(0).get_state().rotation)
for i in range(6):
    pre = np.array(emb.sim.get_agent(0).get_state().position, copy=True)
    emb.sim.step("move_forward")
    post = np.array(emb.sim.get_agent(0).get_state().position, copy=True)
    d = np.linalg.norm(post - pre)
    print(f"Substep {i}: d={d:.4f}, post={post}")
emb.close()

