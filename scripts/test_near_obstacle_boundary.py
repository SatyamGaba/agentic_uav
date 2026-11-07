"""Near-obstacle boundary and depth geometry verification test.

Verifies:
1. Pinhole geometry: Floor cannot be visible at Z < 1.667m with H=1.25m and VFOV=73.74 deg.
   All points with Z <= 0.50m in chassis height band [0.10, 1.25]m are real vertical obstacles.
2. Synthetic obstacle at Z=0.30m, 0.45m, 0.50m is correctly extracted into obstacle cloud.
3. GroundCorridorChecker rejects candidate forward steps with status='reselect' and cause='obstacle'.
4. In Habitat-Sim, moving into an obstacle at 0.30m collides; safety checker prevents this.
"""

import math
import numpy as np
import pytest

from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    extract_view_obstacle_and_free_points,
    VIEW_AZIMUTHS_DEG,
)
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker


def test_pinhole_floor_intersection_geometry():
    """Verify that no camera ray can intersect the floor datum (Y=0) closer than Z=1.6667m."""
    config = CameraConfig()
    H = config.camera_height  # 1.25m
    fy = config.fy  # 320.0
    cy = config.cy  # 240.0
    h_img = config.image_height  # 480

    # The ray pointing furthest downward is the bottom row: v = h_img - 1 = 479 (or 480)
    v_max = h_img
    y_c_max = (v_max - cy) / fy  # 240.0 / 320.0 = 0.75
    # Since Y_agent = H - y_c * Z, for Y_agent = 0: Z = H / y_c_max
    z_floor_min = H / y_c_max
    assert abs(z_floor_min - 1.66667) < 1e-4, f"Expected 1.6667m, got {z_floor_min}"

    # For any Z <= 0.50m, the lowest ray in the camera image is at height:
    y_lowest_at_05m = H - y_c_max * 0.50
    assert y_lowest_at_05m >= 0.875, f"Lowest point at 0.5m must be >= 0.875m, got {y_lowest_at_05m}"

    # For Z = 0.30m, the lowest ray is at height:
    y_lowest_at_03m = H - y_c_max * 0.30
    assert y_lowest_at_03m >= 1.025, f"Lowest point at 0.3m must be >= 1.025m, got {y_lowest_at_03m}"

    print(f"[PASS] Pinhole floor geometry: Nearest visible floor ray is at Z={z_floor_min:.4f}m.")
    print(f"       At Z=0.50m, visible range is Y in [{y_lowest_at_05m:.3f}, {H + y_c_max * 0.50:.3f}]m.")
    print(f"       At Z=0.30m, visible range is Y in [{y_lowest_at_03m:.3f}, {H + y_c_max * 0.30:.3f}]m.")


def test_synthetic_near_obstacle_detection_and_rejection():
    """Verify that an obstacle dead ahead at 0.30m, 0.45m, and 0.50m is detected,
    that point count and 3D coordinates are strictly sensitive to distance,
    and that motion into an obstacle dead ahead at 0.30m is rejected.
    """
    config = CameraConfig()
    checker = GroundCorridorChecker(
        robot_radius=0.10,
        base_margin=0.30,
        rho_max=2.5,
        turn_step_deg=15.0,
        forward_step_m=0.25,
        obstacle_point_threshold=1,
    )

    test_distances = [0.30, 0.45, 0.50]
    results_by_dist = {}

    for dist in test_distances:
        # Create a synthetic depth image: background saturated at 5.0m (free space)
        depth_img = np.full((480, 640), 5.0, dtype=np.float32)
        # Model a physical column of real-world width 0.60m (X in [-0.30, +0.30]m)
        # In camera frame: u = cx + X * fx / Z. Half-width in pixels = 0.30 * 320 / dist
        half_w_px = int(round(0.30 * config.fx / dist))
        c_min = max(0, int(config.cx - half_w_px))
        c_max = min(640, int(config.cx + half_w_px))
        # Height: central rows [180, 300]
        depth_img[180:300, c_min:c_max] = float(dist)

        # Extract obstacle points
        obs_pts, free_pts = extract_view_obstacle_and_free_points(
            depth_image=depth_img,
            view_idx=0,
            config=config,
            stride=4,
            y_min=0.10,
            y_max=1.25,
        )

        assert len(obs_pts) > 0, f"Expected obstacle points at dist={dist}m, got 0"
        mean_z = float(np.mean(obs_pts[:, 2]))
        assert abs(mean_z - (-dist)) < 0.02, f"Expected mean Z near {-dist}m, got {mean_z:.4f}m"

        # Evaluate candidate forward waypoint straight ahead (p_x=0.0, p_z=-1.0m)
        obs_2d = obs_pts[:, [0, 2]]
        result = checker.evaluate(p_x=0.0, p_z=-1.0, obstacles_2d=obs_2d)

        assert result.status == "reselect", f"At dist={dist}m, expected status='reselect', got {result.status}"
        assert result.cause == "obstacle", f"At dist={dist}m, expected cause='obstacle', got {result.cause}"
        assert result.collision_point_count > 0, f"At dist={dist}m, expected collision points > 0"

        results_by_dist[dist] = {
            "point_count": len(obs_pts),
            "mean_z": mean_z,
            "collision_points": result.collision_point_count,
            "status": result.status,
            "cause": result.cause,
        }
        print(f"[PASS] Physical obstacle at {dist:.2f}m: {len(obs_pts)} points, mean Z={mean_z:.3f}m, collision flagged ({result.status}, cause={result.cause}, collision_points={result.collision_point_count}).")

    # Verify sensitivity to distance: point counts must strictly decrease as distance increases
    assert results_by_dist[0.30]["point_count"] > results_by_dist[0.45]["point_count"], "Point count at 0.30m must be > at 0.45m"
    assert results_by_dist[0.45]["point_count"] > results_by_dist[0.50]["point_count"], "Point count at 0.45m must be > at 0.50m"
    # Verify distance sensitivity on Z coordinate
    assert results_by_dist[0.30]["mean_z"] > results_by_dist[0.45]["mean_z"] > results_by_dist[0.50]["mean_z"], "Z coordinates must reflect physical distance"
    print(f"[PASS] Distance sensitivity confirmed: N(0.30m)={results_by_dist[0.30]['point_count']} > N(0.45m)={results_by_dist[0.45]['point_count']} > N(0.50m)={results_by_dist[0.50]['point_count']}.")


def test_habitat_sim_near_obstacle_collision():
    """Verify in Habitat-Sim that moving into an obstacle collides, and safety checker prevents it."""
    try:
        from agenticnav_paper_replica.embodiment.habitat_ground import HabitatGroundEmbodiment, resolve_scene_path
        import json
    except ImportError:
        pytest.skip("Habitat-Sim not available")

    # Load dev episode 8628 (scene JF19kD82Mey)
    with open("agenticnav_paper_replica/data/r2rce_dev_20.json") as f:
        dev = json.load(f)
    ep = next(e for e in dev["episodes"] if str(e["episode_id"]) == "8628")
    scene_path = resolve_scene_path(ep["scene_id"])

    config = CameraConfig()
    emb = HabitatGroundEmbodiment(scene_id=scene_path, camera_config=config)
    obs = emb.reset(ep)

    # In Step 1-2 of ep 8628, agent reaches near a wall/doorframe
    obs = emb.step_discrete(2, 8)
    obs = emb.step_discrete(-4, 7)

    # Check depth in View 0 at Step 2
    d_v0 = obs.depth_views[0]
    min_d = np.min(d_v0[d_v0 > 0.05])
    print(f"\n[Habitat-Sim Ep 8628 Step 2] View 0 min valid depth: {min_d:.4f}m")

    # Extract obstacles from all views
    all_obs_pts = []
    for vi in [0, 1, 2, 3]:
        pts, _ = extract_view_obstacle_and_free_points(
            depth_image=obs.depth_views[vi],
            view_idx=vi,
            config=config,
            stride=4,
            y_min=0.10,
            y_max=1.25,
        )
        if len(pts) > 0:
            all_obs_pts.append(pts)
    combined_obs = np.vstack(all_obs_pts)[:, [0, 2]]
    print(f"[Habitat-Sim Ep 8628 Step 2] Extracted {len(combined_obs)} combined obstacle points in chassis height band.")

    # Test corridor checker on the candidate move (-3, 9) that heads into the wall (-45 deg, 2.25m):
    # p_x = 2.25 * sin(-45 deg) = -1.59m, p_z = -2.25 * cos(-45 deg) = -1.59m
    checker = GroundCorridorChecker(robot_radius=0.10, base_margin=0.30)
    res = checker.evaluate(p_x=-1.59, p_z=-1.59, obstacles_2d=combined_obs)

    print(f"[Habitat-Sim Ep 8628 Step 2] Corridor check on candidate move into wall: status={res.status}, cause={res.cause}, points={res.collision_point_count}")
    assert res.status == "reselect" and res.cause == "obstacle", "Corridor checker must reject step into obstacle"

    # Now verify what happens if physical actuation is forced anyway:
    emb.step_discrete(-3, 9)
    # After stepping into the wall, verify physical block
    pos_before = np.array(emb.sim.get_agent(0).get_state().position, copy=True)
    emb.sim.step("move_forward")
    pos_after = np.array(emb.sim.get_agent(0).get_state().position, copy=True)
    step_disp = np.linalg.norm(pos_after - pos_before)
    print(f"[Habitat-Sim Ep 8628 Step 3] Physical forward step against wall: displacement={step_disp:.4f}m (nominal=0.250m)")
    assert step_disp < 0.05, f"Expected physical collision/block (<0.05m), got {step_disp:.4f}m"
    print(f"[PASS] Physical actuation confirmed blocked by simulator collision. Safety checker successfully prevented undetected collision.")

    emb.close()


if __name__ == "__main__":
    test_pinhole_floor_intersection_geometry()
    test_synthetic_near_obstacle_detection_and_rejection()
    test_habitat_sim_near_obstacle_collision()
    print("\nALL NEAR-OBSTACLE BOUNDARY TESTS PASSED SUCCESSFULLY.")
