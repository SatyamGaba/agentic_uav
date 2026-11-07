"""
Comprehensive Validation & Test Suite for Aerial VLN Adapter Framework.
Tests:
1. Registries & Factory instantiation
2. 2D-to-3D Geometry, camera intrinsics, depth patch filtering, and kinematics
3. Metric calculations: SR, OSR, NE, SPL, Wilson 95% CI
4. Swappable Adapters: AgenticNav, DirectPixel, DiscreteAction, WaypointCandidate, Heuristic
5. End-to-end benchmark execution on MockSimulatorEnv
"""
import sys
import os
import math
import unittest
import numpy as np

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from framework import (
    Pose3D,
    VLNObservation,
    NavAction,
    EpisodeGoal,
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame,
    propagate_body_to_world,
    euclidean_distance_3d,
    evaluate_episode,
    summarize_benchmark,
    compute_wilson_score_interval,
    MockVLMBackend,
    HeuristicRuleAdapter,
    DiscreteActionAdapter,
    DirectPixelAdapter,
    AgenticNavAdapter,
    WaypointCandidateAdapter,
    MockSimulatorEnv,
    VLNBenchmarkRunner,
    create_adapter,
    create_backend,
    create_environment,
    ADAPTER_REGISTRY,
    BACKEND_REGISTRY,
    ENV_REGISTRY
)


class TestVLNFramework(unittest.TestCase):
    """Unit and Integration tests for all components of the VLN pipeline."""

    def test_01_registries_and_factories(self):
        """Validates that all components are properly registered and instantiable via factory."""
        self.assertIn("agentic_nav", ADAPTER_REGISTRY)
        self.assertIn("direct_pixel", ADAPTER_REGISTRY)
        self.assertIn("discrete_action", ADAPTER_REGISTRY)
        self.assertIn("heuristic", ADAPTER_REGISTRY)

        mock_backend = create_backend("mock", model_name="test-model")
        self.assertIsInstance(mock_backend, MockVLMBackend)

        adapter_tool = create_adapter("agentic_nav", backend=mock_backend)
        self.assertIsInstance(adapter_tool, AgenticNavAdapter)

        adapter_pixel = create_adapter("direct_pixel", backend=mock_backend)
        self.assertIsInstance(adapter_pixel, DirectPixelAdapter)

        adapter_discrete = create_adapter("discrete_action", backend=mock_backend)
        self.assertIsInstance(adapter_discrete, DiscreteActionAdapter)

        adapter_heur = create_adapter("heuristic")
        self.assertIsInstance(adapter_heur, HeuristicRuleAdapter)

        env = create_environment("mock")
        self.assertIsInstance(env, MockSimulatorEnv)

    def test_02_camera_intrinsics_and_backprojection(self):
        """Validates pinhole camera intrinsics and 3D back-projection geometry."""
        cam = CameraIntrinsics(width=1920, height=1080, fov_deg=90.0)
        self.assertAlmostEqual(cam.cx, 960.0)
        self.assertAlmostEqual(cam.cy, 540.0)
        # For 90 deg horizontal FOV, fx = 960 / tan(45 deg) = 960.0
        self.assertAlmostEqual(cam.fx, 960.0, places=3)

        # Center pixel (u=0.5, v=0.5) with depth=10.0m should point straight forward
        dx, dy, dz, yaw_offset = backproject_pixel_to_body_frame(
            u=0.5, v=0.5, depth_m=10.0, intrinsics=cam, max_range_m=8.0
        )
        self.assertAlmostEqual(yaw_offset, 0.0, places=3)
        self.assertAlmostEqual(dy, 0.0, places=3)
        self.assertAlmostEqual(dz, 0.0, places=3)
        self.assertAlmostEqual(dx, 8.0, places=3)  # Clamped to max_range_m

        # Right-side pixel (u=0.8, v=0.5) should yield positive dy (right) and positive yaw offset
        dx_r, dy_r, dz_r, yaw_r = backproject_pixel_to_body_frame(
            u=0.8, v=0.5, depth_m=10.0, intrinsics=cam, max_range_m=8.0
        )
        self.assertGreater(yaw_r, 0.0)
        self.assertGreater(dy_r, 0.0)

    def test_03_depth_patch_filtering_and_resolution_decoupling(self):
        """Validates robust metric depth query with sensor holes, NaNs, and resolution mismatch."""
        # Create a 480x640 depth map
        depth_map = np.full((480, 640), 12.0, dtype=np.float32)
        # Introduce a localized region of 8.5m with sensor holes (NaN and 0.0) at the center
        depth_map[238:243, 318:323] = 8.5  # Valid surrounding patch
        depth_map[240, 320] = np.nan
        depth_map[241, 320] = 0.0

        # Query normalized center (u=0.5, v=0.5)
        filtered_depth = query_metric_depth_patch(depth_map, 0.5, 0.5)
        # Median filter over the patch should ignore NaN and 0.0, returning 8.5
        self.assertAlmostEqual(filtered_depth, 8.5, places=2)

    def test_04_world_kinematics_propagation(self):
        """Validates body-to-world coordinate propagation."""
        # Start at (0, 0, 0), facing North (yaw=0.0)
        # Move forward 10m
        x1, y1, z1, yaw1 = propagate_body_to_world(
            current_x=0.0, current_y=0.0, current_z=0.0, current_yaw_rad=0.0,
            dx_body=10.0, dy_body=0.0, dz_body=0.0, yaw_delta_rad=0.0
        )
        self.assertAlmostEqual(x1, 10.0, places=3)
        self.assertAlmostEqual(y1, 0.0, places=3)
        self.assertAlmostEqual(z1, 0.0, places=3)

        # Move forward 10m while turning 90 degrees right (+pi/2)
        x2, y2, z2, yaw2 = propagate_body_to_world(
            current_x=x1, current_y=y1, current_z=z1, current_yaw_rad=yaw1,
            dx_body=10.0, dy_body=0.0, dz_body=2.0, yaw_delta_rad=math.pi / 2.0
        )
        self.assertAlmostEqual(yaw2, math.pi / 2.0, places=3)
        self.assertAlmostEqual(z2, 2.0, places=3)

    def test_05_vln_metrics_and_wilson_ci(self):
        """Validates SR, OSR, NE, SPL, and Wilson Score intervals."""
        # 1. Successful episode (final pos within 5m, optimal path)
        gt_traj = [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [100.0, 0.0, 0.0]]
        agent_traj = [[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [98.0, 1.0, 0.0]]
        goal_pos = [100.0, 0.0, 0.0]

        res = evaluate_episode(
            episode_id=1,
            instruction="Fly straight to landmark",
            trajectory=agent_traj,
            goal_pos=goal_pos,
            gt_trajectory=gt_traj,
            success_radius_m=5.0
        )
        self.assertTrue(res.success)
        self.assertTrue(res.oracle_success)
        self.assertLess(res.navigation_error, 5.0)
        self.assertGreater(res.spl, 0.95)

        # 2. Failed episode
        agent_traj_fail = [[0.0, 0.0, 0.0], [20.0, 0.0, 0.0]]
        res_fail = evaluate_episode(
            episode_id=2,
            instruction="Fly straight to landmark",
            trajectory=agent_traj_fail,
            goal_pos=goal_pos,
            gt_trajectory=gt_traj,
            success_radius_m=5.0
        )
        self.assertFalse(res_fail.success)
        self.assertEqual(res_fail.spl, 0.0)

        # 3. Wilson Score Interval for 16/20 successes (80%)
        ci_lower, ci_upper = compute_wilson_score_interval(16, 20)
        self.assertGreater(ci_lower, 55.0)
        self.assertLess(ci_upper, 95.0)

    def test_06_end_to_end_swappable_adapters(self):
        """Executes full end-to-end benchmark runs with all swappable adapters."""
        env = MockSimulatorEnv()
        mock_backend = MockVLMBackend(mode="tool_call")

        test_goals = [
            EpisodeGoal(
                episode_id=101,
                instruction="Proceed toward the building then turn right to the target",
                goal_pos=[30.0, 10.0, 5.0],
                gt_trajectory=[[0.0, 0.0, 5.0], [20.0, 0.0, 5.0], [30.0, 10.0, 5.0]],
                success_distance=5.0,
                max_steps=10
            ),
            EpisodeGoal(
                episode_id=102,
                instruction="Advance straight forward to the landmark",
                goal_pos=[25.0, 0.0, 5.0],
                gt_trajectory=[[0.0, 0.0, 5.0], [25.0, 0.0, 5.0]],
                success_distance=5.0,
                max_steps=8
            )
        ]

        adapters_to_test = [
            ("AgenticNav (Tool Calling)", create_adapter("agentic_nav", backend=mock_backend)),
            ("DirectPixel (See-Point-Fly)", create_adapter("direct_pixel", backend=MockVLMBackend(mode="direct_pixel"))),
            ("DiscreteAction (OpenFly)", create_adapter("discrete_action", backend=MockVLMBackend(mode="discrete_actions"))),
            ("WaypointCandidate (Candidate Ranking)", create_adapter("waypoint_candidate", backend=mock_backend)),
            ("Heuristic Baseline", create_adapter("heuristic"))
        ]

        for adapter_name, adapter in adapters_to_test:
            runner = VLNBenchmarkRunner(
                adapter=adapter,
                env=env,
                model_name="mock-qwen-vl",
                max_steps_per_episode=10,
                success_radius_m=5.0
            )
            summary = runner.run_benchmark(test_goals)
            self.assertEqual(summary.total_episodes, len(test_goals))
            self.assertGreaterEqual(summary.mean_pred_length, 0.0)
            self.assertGreaterEqual(summary.success_rate, 0.0)
            print(f"Verified adapter: {adapter_name} successfully executed {summary.total_episodes} episodes.")


if __name__ == "__main__":
    unittest.main()
