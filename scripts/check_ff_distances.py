import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.geometry.pinhole import CameraConfig
from scripts.sweep_g64b_multiscene import select_poses_across_scenes, extract_all_valid_agent_points
from habitat_sim.utils.common import quat_from_angle_axis, quat_rotate_vector

config = ReplicaConfig()
camera_cfg = CameraConfig(
    image_width=config.image_width,
    image_height=config.image_height,
    hfov_deg=config.hfov_deg,
    d_min=config.d_min,
    d_max=config.d_max,
    camera_height=config.camera_height,
)

poses = select_poses_across_scenes(n_per_scene=5)
headings_deg = [-135, -90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90, 135, 180]
distances_m = [0.5, 1.0, 1.5, 2.0, 2.5]

r_a = 0.18
y_min = 0.15
y_max = 1.25

ff_recs = []
current_scene = None
emb = None

for p_idx, p in enumerate(poses):
    scene_name = p["scene_id"]
    if scene_name != current_scene:
        if emb is not None:
            emb.close()
        current_scene = scene_name
        emb = HabitatGroundEmbodiment(scene_id=scene_name, camera_config=camera_cfg, gpu_device_id=-1, max_actions=50)

    obs = emb.reset(p)
    depth_views = obs.depth_views
    sim = emb.sim
    agent = sim.get_agent(0)
    pf = sim.pathfinder
    init_agent_state = agent.get_state()

    all_pts = extract_all_valid_agent_points(depth_views, camera_cfg, stride=4, eps=1e-3)

    for th in headings_deg:
        th_rad = math.radians(th)
        sin_th = math.sin(th_rad)
        cos_th = math.cos(th_rad)

        ox = all_pts[:, 0]
        oy = all_pts[:, 1]
        oz = all_pts[:, 2]

        s = -ox * sin_th - oz * cos_th
        h = ox * cos_th - oz * sin_th

        rot_m = quat_from_angle_axis(th_rad, np.array([0.0, 1.0, 0.0]))
        agent_rot = init_agent_state.rotation * rot_m
        forward_vec = quat_rotate_vector(agent_rot, np.array([0.0, 0.0, -1.0]))

        for rho in distances_m:
            cur_pos = np.array(init_agent_state.position, copy=True)
            start_p = np.array(cur_pos, copy=True)
            n_forward = int(round(rho / 0.25))
            sim_collided = False
            for _ in range(n_forward):
                next_target = cur_pos + forward_vec * 0.25
                stepped_pos = pf.try_step(cur_pos, next_target)
                sub_dist = float(np.linalg.norm(stepped_pos - cur_pos))
                if sub_dist < 0.05:
                    sim_collided = True
                cur_pos = stepped_pos

            dist_moved = float(np.linalg.norm(cur_pos - start_p))
            dist_deficit = rho - dist_moved
            sim_blocked = sim_collided or (dist_deficit > 0.20)

            in_body_corr = (s > 0.0) & (s < (rho + 0.10)) & (np.abs(h) < 0.10) & (oy >= 0.15) & (oy <= 0.80)
            body_pts = int(np.sum(in_body_corr))
            truth_blocked = bool(sim_blocked or (body_pts >= 5))

            in_corr = (s > 0.0) & (s < (rho + r_a)) & (np.abs(h) < r_a) & (oy >= y_min) & (oy <= y_max)
            checker_blocked = bool(np.any(in_corr))

            if truth_blocked and not checker_blocked:
                ff_recs.append({
                    "pose": p_idx,
                    "scene": scene_name,
                    "th": th,
                    "rho": rho,
                    "dist_moved": round(dist_moved, 2),
                    "dist_deficit": round(dist_deficit, 2),
                    "sim_coll": sim_collided,
                    "body_pts": body_pts,
                })

if emb is not None:
    emb.close()

print(f"Total FF cases: {len(ff_recs)}")
cnt_rho = Counter(r["rho"] for r in ff_recs)
for rho, c in sorted(cnt_rho.items()):
    print(f"  rho={rho}m: {c} cases")

print("\nSample 10 FF cases:")
for r in ff_recs[:10]:
    print(f"  Pose {r['pose']} ({r['scene']}), th={r['th']}°, rho={r['rho']}m -> moved={r['dist_moved']}m, def={r['dist_deficit']}m, coll={r['sim_coll']}, body_pts={r['body_pts']}")

