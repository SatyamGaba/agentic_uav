import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, extract_view_obstacle_and_free_points
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "1868")

camera_cfg = CameraConfig()
emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)

checker = GroundCorridorChecker(robot_radius=0.10, obstacle_point_threshold=5)
all_obs = []
for v_i in range(4):
    depth_m = obs.depth_views[v_i]
    obs_pts, _ = extract_view_obstacle_and_free_points(depth_m, v_i, camera_cfg)
    if len(obs_pts) > 0:
        all_obs.append(obs_pts[:, [0, 2]])
obs_2d = np.concatenate(all_obs, axis=0) if all_obs else np.empty((0, 2))

cache = EpisodicImageCache()
executor = ToolExecutor(camera_cfg, checker, cache)

# Test moves in View 3 with various u, v
for u in [0.4, 0.5, 0.6, 0.7]:
    for v in [0.70, 0.75, 0.80, 0.85]:
        res = executor.execute_move_to(
            view=3, u=u, v=v, reason="test",
            depth_views=obs.depth_views,
            obstacle_points_agent_xz=obs_2d
        )
        print(f"View 3 (u={u:.2f}, v={v:.2f}) -> status={res.get('status')}, theta={res.get('theta_deg')}, rho={res.get('rho_m')}, cause={res.get('cause')}")

emb.close()

