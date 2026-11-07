import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

sim = emb.sim
agent = sim.get_agent(0)
st = habitat_sim.AgentState()
st.position = np.array([-1.78, 3.3189, 3.82], dtype=np.float32)
half_yaw = 1.05 / 2.0
st.rotation = np.quaternion(math.cos(half_yaw), 0, math.sin(half_yaw), 0)
agent.set_state(st)
obs = emb._extract_obs()

d1 = obs.depth_views[1]
print("View 1 shape:", d1.shape)
print("View 1 min:", d1.min(), "max:", d1.max())

# Sample depths across the bottom 100 rows: y=380..479
print("Depths along y=420 (x=40..600, step 40):")
print([round(float(d1[420, x]), 2) for x in range(40, 640, 40)])
print("Depths along y=450 (x=40..600, step 40):")
print([round(float(d1[450, x]), 2) for x in range(40, 640, 40)])
print("Depths along y=470 (x=40..600, step 40):")
print([round(float(d1[470, x]), 2) for x in range(40, 640, 40)])

emb.close()

