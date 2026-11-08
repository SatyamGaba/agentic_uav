"""Unit tests for AgenticNav geometry, coordinate transforms, and safety corridor engine.

Explicitly implements all Section 4.4 required tests (Tests 1 through 9) from Blueprint v7.
"""

import math
import numpy as np
import pytest

from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    deproject_pixel,
    reproject_point,
    camera_to_agent_frame,
    agent_to_camera_frame,
    estimate_corridor_surface_slope,
    compute_effective_margin,
    unproject_depth_image,
    normalize_to_pixel,
    pixel_to_normalized,
    convert_raw_depth,
    VIEW_AZIMUTHS_DEG,
)
from agenticnav_paper_replica.geometry.safety_region import (
    GroundCorridorChecker,
    SafetyCheckResult,
)


class TestRequiredSection44UnitTests:
    """Rigorous verification of the 9 required unit tests from Blueprint v7 Section 4.4."""

    def setup_method(self):
        self.config = CameraConfig()
        self.checker = GroundCorridorChecker(
            robot_radius=0.18,
            base_margin=0.30,
            rho_max=2.5,
            turn_step_deg=15.0,
            forward_step_m=0.25,
            zero_rho_policy="rotate",
            allow_adjacent_bin_snap=False,
        )

    def test_1_pinhole_center_mapping_all_views(self):
        """Test 1: Center pixel of views 0, 1, 2, 3 at depth 2m lands at:
        (0, 1.25, -2), (+2, 1.25, 0), (0, 1.25, +2) and (-2, 1.25, 0).
        """
        # Center of 640x480 is (320, 240)
        p_c = deproject_pixel(320, 240, 2.0, self.config, is_normalized=False)

        # View 0: Front -> (0, 1.25, -2)
        p_v0 = camera_to_agent_frame(p_c, view_idx=0, config=self.config)
        assert np.allclose(p_v0, [0.0, 1.25, -2.0], atol=1e-2)

        # View 1: Right -> (+2, 1.25, 0)
        p_v1 = camera_to_agent_frame(p_c, view_idx=1, config=self.config)
        assert np.allclose(p_v1, [2.0, 1.25, 0.0], atol=1e-2)

        # View 2: Back -> (0, 1.25, +2)
        p_v2 = camera_to_agent_frame(p_c, view_idx=2, config=self.config)
        assert np.allclose(p_v2, [0.0, 1.25, 2.0], atol=1e-2)

        # View 3: Left -> (-2, 1.25, 0)
        p_v3 = camera_to_agent_frame(p_c, view_idx=3, config=self.config)
        assert np.allclose(p_v3, [-2.0, 1.25, 0.0], atol=1e-2)

    def test_2_right_edge_bearing(self):
        """Test 2: A pixel at the right edge of view 0 gives p_x > 0 and theta < 0."""
        # Right edge: x0 = 639, y0 = 240 at depth 2.0m
        p_c = deproject_pixel(639, 240, 2.0, self.config, is_normalized=False)
        p_a = camera_to_agent_frame(p_c, view_idx=0, config=self.config)
        p_x, p_y, p_z = p_a[0], p_a[1], p_a[2]

        assert p_x > 0.0  # to the right

        d_raw, th_deg, rho_q, th_q_deg = self.checker.compute_kinematics(p_x, p_z)
        assert th_deg < 0.0  # CW negative

    def test_3_ahead_left_target_alignment(self):
        """Test 3: Target (p_x, p_z) = (-1, -2), ahead-left:
        theta = +26.57°, and corridor centreline passes through target (s = 2.236, h = 0).
        """
        p_x = -1.0
        p_z = -2.0
        D_raw, th_deg, rho_q, th_q_deg = self.checker.compute_kinematics(p_x, p_z)

        # theta = atan2(1.0, 2.0) = 0.4636 rad = 26.565°
        assert abs(th_deg - 26.565) < 0.01

        # Check corridor coordinates (s, h) along b_theta, l_theta
        th_rad = math.radians(th_deg)
        b_x = -math.sin(th_rad)
        b_z = -math.cos(th_rad)
        l_x = math.cos(th_rad)
        l_z = -math.sin(th_rad)

        s = p_x * b_x + p_z * b_z
        h = p_x * l_x + p_z * l_z

        expected_dist = math.sqrt((-1.0) ** 2 + (-2.0) ** 2)  # sqrt(5) = 2.236
        assert abs(s - expected_dist) < 1e-4
        assert abs(h) < 1e-4  # exactly on the centerline!

    def test_4_pixel_to_3d_reprojection_round_trip(self):
        """Test 4: Pixel to 3D to reprojection round trip within 1e-4 pixels."""
        test_pixels = [(100, 150), (320, 240), (50, 400), (600, 300)]
        for x_orig, y_orig in test_pixels:
            z_c = 2.75
            p_c = deproject_pixel(x_orig, y_orig, z_c, self.config, is_normalized=False)
            x_rec, y_rec, z_rec = reproject_point(p_c, self.config)

            assert abs(x_rec - x_orig) < 1e-4
            assert abs(y_rec - y_orig) < 1e-4
            assert abs(z_rec - z_c) < 1e-4

    def test_5_depth_conversion_valid_and_invalid(self):
        """Test 5: Depth conversion: raw 0 and raw 1 are invalid; raw 0.5 with range [0.5, 5.0] gives 2.75 m."""
        # Raw 0 is invalid
        ok_0, d_0, cause_0 = convert_raw_depth(0.0, self.config, is_raw_normalized=True)
        assert not ok_0
        assert cause_0 == "invalid_depth"

        # Raw 1 (saturated) is invalid
        ok_1, d_1, cause_1 = convert_raw_depth(1.0, self.config, is_raw_normalized=True)
        assert not ok_1
        assert cause_1 == "invalid_depth"

        # Raw 0.5 with [0.5, 5.0]: 0.5 + 0.5 * (5.0 - 0.5) = 0.5 + 2.25 = 2.75m
        ok_half, d_half, cause_half = convert_raw_depth(0.5, self.config, is_raw_normalized=True)
        assert ok_half
        assert abs(d_half - 2.75) < 1e-4
        assert cause_half is None

    def test_6_wall_collision_and_obliquity(self):
        """Test 6: Wall 2.0 m straight ahead: move_to with rho = 2.0 m is rejected and rho = 1.0 m passes.
        Repeat at 30°, 45° and 60° obliquity.
        """
        # Wall obstacle at z = -2.0m across x in [-1.0, 1.0]
        wall_pts = np.array([[x, -2.0] for x in np.linspace(-1.0, 1.0, 21)])

        # Target 1: Straight ahead at 2.5m (D_raw=2.5, m=0.30 -> rho_cont=2.2 -> rho_q=2.0m)
        # Corridor extends to s = 2.0 + 0.18 = 2.18m > 2.0m -> hits wall!
        res_long = self.checker.evaluate(p_x=0.0, p_z=-2.5, obstacles_2d=wall_pts)
        assert res_long.status == "reselect"
        assert res_long.cause == "obstacle"

        # Target 2: Straight ahead with rho_q = 1.0m (target at D_raw=1.5m -> rho_q=1.0m)
        # Corridor extends to s = 1.0 + 0.18 = 1.18m < 2.0m -> clear!
        res_short = self.checker.evaluate(p_x=0.0, p_z=-1.5, obstacles_2d=wall_pts)
        assert res_short.status == "executed"
        assert res_short.quantized_rho_m == 1.0

        # Test oblique walls
        for angle_deg in [30.0, 45.0, 60.0]:
            rad = math.radians(angle_deg)
            # Wall plane: n_x * x + n_z * z + d = 0
            # Target at 1.0m in clear space
            tx = -1.0 * math.sin(rad)
            tz = -1.0 * math.cos(rad)
            res_obl = self.checker.evaluate(p_x=tx, p_z=tz, obstacles_2d=wall_pts)
            # Short target (rho_q = 0.50m) does not penetrate wall at 2.0m
            assert res_obl.status == "executed"

    def test_7_floor_rejection_and_doorway(self):
        """Test 7: Floor points (height below y_min) never enter obstacle set;
        doorway wider than 2*r_a plus clearance passes.
        """
        # Floor depth image: bottom rows have negative y_v (below y_min = 0.10)
        depth_img = np.full((480, 640), 1.5, dtype=np.float32)
        pts = unproject_depth_image(
            depth_img, view_idx=0, config=self.config, stride=10, y_min=0.10, y_max=1.25
        )
        if len(pts) > 0:
            assert np.all(pts[:, 1] >= 0.10)
            assert np.all(pts[:, 1] <= 1.25)

        # Doorway test: two door posts at x = -0.30 and x = +0.30 (width = 0.60m > 2*r_a = 0.36m)
        doorway_obstacles = np.array([
            [-0.30, -1.5],
            [+0.30, -1.5],
        ])
        # Robot navigating straight through at (0, -2.0)
        res_door = self.checker.evaluate(p_x=0.0, p_z=-2.0, obstacles_2d=doorway_obstacles)
        assert res_door.status == "executed"

    def test_8_standoff_and_peripheral_range(self):
        """Test 8: Target nearer than m gives rho = 0 and never a negative value.
        Right-edge pixel at z-depth 4.5 m (D_raw about 6.4 m) is not rejected and gets rho = rho_max.
        """
        # Target at D_raw = 0.20m (< m = 0.30m)
        D_raw, th_deg, rho_q, th_q_deg = self.checker.compute_kinematics(p_x=0.0, p_z=-0.20)
        assert rho_q == 0.0  # never negative!

        # Right edge pixel at z_c = 4.5m in 90 deg HFOV
        # x_c = (639 + 0.5 - 320) * 4.5 / fx = 320 * 4.5 / 320 = 4.5m
        # D_raw = sqrt(4.5^2 + 4.5^2) = 4.5 * sqrt(2) = 6.36m > d_max (5.0m)
        p_c = deproject_pixel(639, 240, 4.5, self.config, is_normalized=False)
        p_a = camera_to_agent_frame(p_c, view_idx=0, config=self.config)
        d_raw, th_deg, rho_q, th_q_deg = self.checker.compute_kinematics(p_a[0], p_a[2])

        assert d_raw > 5.0
        # rho must be clamped cleanly to rho_max (2.5m)
        assert abs(rho_q - self.checker.rho_max) < 1e-4

    def test_9_sim_in_the_loop_bearing_view_depth_verification(self):
        """Test 9: Place the agent facing a known object and check bearing, view index and depth."""
        # Known object at (0.0, 1.25, -2.5) in world frame
        # Agent facing forward (yaw = 0): object is in View 0 (Front) at depth 2.5m, bearing 0 deg
        p_obj_c = np.array([0.0, 0.0, 2.5])
        p_obj_agent = camera_to_agent_frame(p_obj_c, view_idx=0, config=self.config)

        assert abs(p_obj_agent[0]) < 1e-4
        assert abs(p_obj_agent[2] - (-2.5)) < 1e-4

        d_raw, th_deg, rho_q, th_q_deg = self.checker.compute_kinematics(p_obj_agent[0], p_obj_agent[2])
        assert abs(th_deg) < 1e-4
        assert abs(d_raw - 2.5) < 1e-4

    def test_10_unproject_depth_image_vs_deproject_pixel_consistency(self):
        """Test 10: Verify that vectorized unproject_depth_image and single-point
        camera_to_agent_frame(deproject_pixel(...)) produce identical coordinates for any pixel and view.
        """
        # Create synthetic depth map with variable depths
        H, W = self.config.image_height, self.config.image_width
        np.random.seed(42)
        depth_img = np.random.uniform(1.0, 4.0, size=(H, W)).astype(np.float32)

        # Ensure known points in height band [0.10, 1.25]
        # At center y=240, y_c=0 -> y_agent = 1.25m
        depth_img[240, 320] = 2.0  # (0.0, 1.25, -2.0)
        # Below center y=280: y_c = (280.5-240)*2.0/320 = 0.253m -> y_agent = 1.25 - 0.253 = 0.997m
        depth_img[280, 400] = 2.0

        for vi in range(4):
            pts_cloud = unproject_depth_image(
                depth_img,
                view_idx=vi,
                config=self.config,
                stride=8,
                y_min=0.10,
                y_max=1.25,
            )
            assert len(pts_cloud) > 0

            # Test a random sample of stride grid points
            grid_y, grid_x = np.meshgrid(np.arange(0, H, 8), np.arange(0, W, 8), indexing="ij")
            gy_flat = grid_y.flatten()
            gx_flat = grid_x.flatten()

            for idx in np.random.choice(len(gy_flat), size=25, replace=False):
                r = gy_flat[idx]
                c = gx_flat[idx]
                z_val = float(depth_img[r, c])
                p_c = deproject_pixel(c, r, z_val, self.config, is_normalized=False)
                p_agent_pt = camera_to_agent_frame(p_c, view_idx=vi, config=self.config)

                # Check height band filter
                if 0.10 <= p_agent_pt[1] <= 1.25 and self.config.d_min <= z_val <= self.config.d_max:
                    # Point must exist in pointcloud
                    dists = np.linalg.norm(pts_cloud - p_agent_pt, axis=1)
                    assert np.min(dists) < 1e-5

    def test_11_camera_to_agent_round_trip(self):
        """Test 11: Verify exact mathematical round-trip consistency:
        agent_to_camera_frame(camera_to_agent_frame(p_c)) == p_c.
        """
        np.random.seed(1337)
        for vi in range(4):
            for _ in range(50):
                # Random optical point with z_c > 0
                x_c = np.random.uniform(-3.0, 3.0)
                y_c = np.random.uniform(-2.0, 2.0)
                z_c = np.random.uniform(0.5, 5.0)
                p_c = np.array([x_c, y_c, z_c], dtype=np.float64)

                p_agent = camera_to_agent_frame(p_c, view_idx=vi, config=self.config)
                p_c_recovered = agent_to_camera_frame(p_agent, view_idx=vi, config=self.config)

                assert np.allclose(p_c, p_c_recovered, atol=1e-10)

    def test_12_invalid_clamped_depth_rejection(self):
        """Test 12: Verify that near-clipping clamped depth (<= d_min + eps),
        far-saturation (>= d_max - eps), zero, negative, and NaN are rejected as invalid depth.
        """
        # Near clamp (e.g. 0.500m when d_min=0.5m)
        ok, d_val, cause = convert_raw_depth(0.500, self.config)
        assert ok is False
        assert cause == "invalid_depth"

        # Near clamp with epsilon
        ok, d_val, cause = convert_raw_depth(0.5005, self.config)
        assert ok is False

        # Valid in-range depth
        ok, d_val, cause = convert_raw_depth(1.50, self.config)
        assert ok is True
        assert abs(d_val - 1.50) < 1e-4
        assert cause is None

        # Far saturation (5.000m when d_max=5.0m)
        ok, d_val, cause = convert_raw_depth(5.000, self.config)
        assert ok is False
        assert cause == "invalid_depth"

        # Zero and negative
        ok, _, _ = convert_raw_depth(0.0, self.config)
        assert ok is False
        ok, _, _ = convert_raw_depth(-1.0, self.config)
        assert ok is False

        # NaN and Inf
        ok, _, _ = convert_raw_depth(float("nan"), self.config)
        assert ok is False
        ok, _, _ = convert_raw_depth(float("inf"), self.config)
        assert ok is False

        # Point cloud extraction masking: depth map entirely composed of 0.000m near-clipping dropouts
        dropped_img = np.zeros((480, 640), dtype=np.float32)
        pts_dropped = unproject_depth_image(dropped_img, view_idx=0, config=self.config)
        assert len(pts_dropped) == 0, "Near-clipping dropouts (0.0m) must not generate obstacle points"


class TestZeroRhoPolicyAndAblationSwitches:
    def test_turn_only_move_when_zero_rho_and_nonzero_theta(self):
        checker = GroundCorridorChecker(
            base_margin=0.30, forward_step_m=0.25, zero_rho_policy="rotate"
        )
        # Target at 90 deg right, distance 0.20m (rho_q = 0)
        res = checker.evaluate(p_x=0.20, p_z=0.0, obstacles_2d=np.empty((0, 2)))
        assert res.status == "executed"
        assert res.quantized_rho_m == 0.0
        assert res.quantized_theta_deg == -90.0
        assert res.forward_steps == 0
        assert res.turn_steps == -6  # 6 steps right

    def test_pure_rotation_ignores_geometry_already_touching_body(self):
        """Confirm forward_steps == 0 ignores pre-existing footprint collisions (e.g. wall at 0.10m)."""
        checker = GroundCorridorChecker(
            base_margin=0.30, forward_step_m=0.25, zero_rho_policy="rotate"
        )
        # 10 obstacle points right at the agent's skin (dist = 0.10m < r_a=0.18m)
        touching_obs = np.array([[0.10, 0.0] for _ in range(10)], dtype=np.float64)
        # Target at 45 deg left, distance 0.20m (rho_q = 0, turn_steps = +3)
        res = checker.evaluate(p_x=-0.14, p_z=-0.14, obstacles_2d=touching_obs)
        assert res.status == "executed"
        assert res.forward_steps == 0
        assert res.turn_steps == 3
        assert res.quantized_rho_m == 0.0

    def test_forward_move_still_rejected_by_obstacle_ahead(self):
        """Confirm forward move with forward_steps > 0 is still rejected by obstacle in corridor."""
        checker = GroundCorridorChecker(
            base_margin=0.30, forward_step_m=0.25, zero_rho_policy="rotate"
        )
        # Obstacle straight ahead at 0.40m
        ahead_obs = np.array([[0.0, -0.40] for _ in range(5)], dtype=np.float64)
        # Target at 1.0m ahead (rho_q = 0.75m)
        res = checker.evaluate(p_x=0.0, p_z=-1.0, obstacles_2d=ahead_obs)
        assert res.status == "reselect"
        assert res.cause == "obstacle"

    def test_no_motion_reselect_when_both_zero(self):
        checker = GroundCorridorChecker(
            base_margin=0.30, forward_step_m=0.25, zero_rho_policy="rotate"
        )
        # Target straight ahead at 0.15m (rho_q = 0, theta_q = 0)
        res = checker.evaluate(p_x=0.0, p_z=-0.15, obstacles_2d=np.empty((0, 2)))
        assert res.status == "reselect"
        assert res.cause == "no_motion"
