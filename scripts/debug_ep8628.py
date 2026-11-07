import json, habitat_sim, math, numpy as np
from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment
from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.geometry.pinhole import CameraConfig, agent_to_camera_frame, reproject_point, pixel_to_normalized, VIEW_AZIMUTHS_DEG
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache
from agenticnav_paper_replica.harness.executor import ToolExecutor

with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
    dev = json.load(f)
ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
camera_cfg = CameraConfig()
checker = GroundCorridorChecker(robot_radius=0.10, base_margin=0.30, rho_max=2.5)
cache = EpisodicImageCache()
executor = ToolExecutor(camera_config=camera_cfg, corridor_checker=checker, image_cache=cache)

emb = HabitatGroundEmbodiment(scene_id=ep["scene_id"], camera_config=camera_cfg)
obs = emb.reset(ep)
sim = emb.sim
pf = sim.pathfinder

for step in range(1, 10):
    agent_state = sim.get_agent(0).get_state()
    pos = agent_state.position
    rot = agent_state.rotation
    yaw = 2.0 * math.atan2(rot.y, rot.w)
    goal_pos = np.array(ep["goals"][0]["position"], dtype=np.float32)

    path = habitat_sim.ShortestPath()
    path.requested_start = pos
    path.requested_end = goal_pos
    found = pf.find_path(path)
    dist = path.geodesic_distance if found else float(np.linalg.norm(goal_pos - pos))

    target_d = min(1.5, dist)
    pts = list(path.points)
    accum = 0.0
    target_pt = pts[-1]
    for i in range(1, len(pts)):
        seg = float(np.linalg.norm(pts[i] - pts[i-1]))
        if accum + seg >= target_d:
            ratio = (target_d - accum) / max(seg, 1e-6)
            target_pt = pts[i-1] + ratio * (pts[i] - pts[i-1])
            break
        accum += seg

    dx = float(target_pt[0] - pos[0])
    dz = float(target_pt[2] - pos[2])
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x_a = cos_y * dx - sin_y * dz
    z_a = sin_y * dx + cos_y * dz
    p_agent = np.array([x_a, 0.0, z_a], dtype=np.float64)

    theta_rad = math.atan2(-x_a, -z_a)
    theta_deg = math.degrees(theta_rad)

    best_view = 0
    min_diff = 999.0
    for v_idx, v_deg in VIEW_AZIMUTHS_DEG.items():
        diff = abs((theta_deg - v_deg + 180.0) % 360.0 - 180.0)
        if diff < min_diff:
            min_diff = diff
            best_view = v_idx

    p_c = agent_to_camera_frame(p_agent, best_view, camera_cfg)
    x0, y0, _ = reproject_point(p_c, camera_cfg)
    u, v = pixel_to_normalized(int(round(x0)), int(round(y0)), camera_cfg)
    u = float(np.clip(u, 0.05, 0.95))
    v = float(np.clip(v, 0.52, 0.95))

    exec_res = executor.execute_move_to(
        view=best_view, u=u, v=v, reason="test",
        depth_views=obs.depth_views,
        obstacle_points_agent_xz=np.empty((0, 2))
    )

    print(f"Step {step}: pos={pos}, dist={dist:.2f}, yaw={yaw:.2f}, target_pt={target_pt}, heading={theta_deg:.1f}, view={best_view}, u={u:.2f}, v={v:.2f}, exec_status={exec_res.get('status')}")
    if exec_res.get("status") == "executed":
        turn_steps = exec_res.get("turn_steps", 0)
        fwd_steps = exec_res.get("forward_steps", 0)
        print(f"  Executing {turn_steps} turns, {fwd_steps} fwds")
        obs = emb.step_discrete(turn_steps_15deg=turn_steps, forward_steps_025m=fwd_steps)
        print(f"  Moved: actual={obs.actual_distance_moved:.2f}m, collided={obs.collided}, new_pos={obs.position}")
    else:
        print(f"  Rejected: cause={exec_res.get('cause')}")
        break

emb.close()
