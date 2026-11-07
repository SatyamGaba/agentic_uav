"""
OpenFly Episode 0 Environment & Observation Contract Audit
==========================================================
Audits the physical contract between OpenFlyBenchmarkEnv and OpenFlyBenchmarkAdapter:
1. Observation contract: RGB image shape/dtype, depth map channel/metrics, coordinate frames.
2. Action execution contract: exact physical displacement of discrete macro-actions (0-9).
3. Closed-loop adapter coupling: verifies the continuous controller drives the benchmark env.
"""
import math
import numpy as np
import pytest

from framework.environments.openfly_env import OpenFlyBenchmarkEnv
from framework.types import NavAction
from framework.adapters.discrete_action import OPENFLY_ACTION_VOCAB
from adapters.openfly_adapter import (
    OpenFlyBenchmarkAdapter,
    ACTION_STOP,
    ACTION_MOVE_FORWARD_FINE,
    ACTION_TURN_RIGHT_30,
    ACTION_TURN_LEFT_30,
    ACTION_CLIMB_3M,
    ACTION_DESCEND_3M,
    ACTION_STRAFE_LEFT_3M,
    ACTION_STRAFE_RIGHT_3M,
    ACTION_MOVE_FORWARD_MED,
    ACTION_MOVE_FORWARD_LARGE,
)

def test_openfly_episode0_observation_contract():
    env = OpenFlyBenchmarkEnv(dataset_path="benchmark/seen.json", enable_visuals=True)
    goal = env.load_episode_goal(0)
    obs = env.reset(goal)

    # 1. Image Resolution and Channels
    assert obs.rgb_image is not None, "RGB image observation must not be None"
    assert obs.rgb_image.shape == (360, 640, 3), f"Unexpected RGB shape: {obs.rgb_image.shape}"
    assert obs.rgb_image.dtype == np.uint8, f"Unexpected RGB dtype: {obs.rgb_image.dtype}"
    assert obs.rgb_image.min() >= 0 and obs.rgb_image.max() <= 255, "RGB values outside [0, 255]"

    # 2. Metric Depth Map
    assert obs.depth_map is not None, "Depth map must not be None"
    assert obs.depth_map.shape == (360, 640), f"Unexpected Depth shape: {obs.depth_map.shape}"
    assert obs.depth_map.dtype == np.float32, f"Unexpected Depth dtype: {obs.depth_map.dtype}"
    assert np.all(obs.depth_map > 0.0), "Depth contains non-positive values"
    assert np.all(~np.isnan(obs.depth_map)), "Depth contains NaN values"
    assert np.all(~np.isinf(obs.depth_map)), "Depth contains Inf values"

    # 3. Telemetry and Pose Coordinate Frame
    assert obs.pose is not None, "Pose must not be None"
    assert math.isclose(obs.pose.yaw, -0.5235987755982988, rel_tol=1e-5), "Start yaw mismatch"
    # Starting pose P0 = Pos[0] - dist * cos(yaw)
    expected_x0 = 270.4203752124709 - 9.0 * math.cos(-0.5235987755982988)
    assert math.isclose(obs.pose.x, expected_x0, rel_tol=1e-5), "Start X position mismatch"


def test_openfly_action_execution_kinematics_audit():
    env = OpenFlyBenchmarkEnv(dataset_path="benchmark/seen.json", enable_visuals=False)
    goal = env.load_episode_goal(0)
    obs0 = env.reset(goal)

    # Step 1: Action 9 (MOVE_FORWARD_LARGE, +9.0m) along initial yaw (-30 deg)
    yaw0 = obs0.pose.yaw
    obs1, done, info = env.step(NavAction(action_code=ACTION_MOVE_FORWARD_LARGE))
    dx1 = obs1.pose.x - obs0.pose.x
    dy1 = obs1.pose.y - obs0.pose.y
    dist1 = math.hypot(dx1, dy1)
    assert math.isclose(dist1, 9.0, abs_tol=1e-4), f"Action 9 displaced {dist1}m, expected 9.0m"
    assert math.isclose(dx1, 9.0 * math.cos(yaw0), abs_tol=1e-4), f"Action 9 dx mismatch: {dx1}"
    assert math.isclose(dy1, 9.0 * math.sin(yaw0), abs_tol=1e-4), f"Action 9 dy mismatch: {dy1}"
    assert math.isclose(obs1.pose.yaw, yaw0, abs_tol=1e-4), "Action 9 should not change yaw"

    # Step 2: Action 3 (TURN_RIGHT, -30 deg CW, verified from seen.json)
    obs2, done, info = env.step(NavAction(action_code=ACTION_TURN_RIGHT_30))
    expected_yaw2 = yaw0 - math.radians(30.0)  # -30 deg - 30 deg = -60 deg
    assert math.isclose(obs2.pose.yaw, expected_yaw2, abs_tol=1e-4), f"Action 3 yaw mismatch: {obs2.pose.yaw}"
    assert math.isclose(obs2.pose.x, obs1.pose.x, abs_tol=1e-4), "Action 3 should not translate X"
    assert math.isclose(obs2.pose.y, obs1.pose.y, abs_tol=1e-4), "Action 3 should not translate Y"

    # Step 3: Action 2 (TURN_LEFT, +30 deg CCW, reverses back to -30 deg)
    obs3, done, info = env.step(NavAction(action_code=ACTION_TURN_LEFT_30))
    assert math.isclose(obs3.pose.yaw, yaw0, abs_tol=1e-4), f"Action 2 yaw mismatch: {obs3.pose.yaw}"

    # Step 4: Action 4 (CLIMB, +3.0m)
    obs4, done, info = env.step(NavAction(action_code=ACTION_CLIMB_3M))
    dz4 = obs4.pose.z - obs3.pose.z
    assert math.isclose(dz4, 3.0, abs_tol=1e-4), f"Action 4 dz mismatch: {dz4}"

    # Step 5: Action 6 (STRAFE_RIGHT, dy = -3.0m starboard)
    obs5, done, info = env.step(NavAction(action_code=ACTION_STRAFE_RIGHT_3M))
    # At yaw = -30 deg, dy_body = -3.0 rotates into world frame:
    # dx_w = -(-3.0)*sin(-30 deg) = 3.0 * (-0.5) = -1.5m
    # dy_w = (-3.0)*cos(-30 deg) = -3.0 * 0.866 = -2.598m
    dx5 = obs5.pose.x - obs4.pose.x
    dy5 = obs5.pose.y - obs4.pose.y
    assert math.isclose(dx5, 1.5 * math.sin(-math.radians(30.0)) * 2, abs_tol=1e-3)
    assert math.isclose(dy5, -3.0 * math.cos(math.radians(30.0)), abs_tol=1e-3)

    # Step 6: Action 0 (STOP)
    obs6, done, info = env.step(NavAction(is_stop=True, action_code=ACTION_STOP))
    assert done is True, "Action STOP must terminate episode"
    assert info["reason"] == "STOP executed", f"Unexpected stop reason: {info['reason']}"


def test_openfly_adapter_closed_loop_coupling():
    env = OpenFlyBenchmarkEnv(dataset_path="benchmark/seen.json", enable_visuals=True)
    goal = env.load_episode_goal(0)
    obs = env.reset(goal)

    # Initialize OpenFlyBenchmarkAdapter with environment's initial state
    adapter = OpenFlyBenchmarkAdapter(
        initial_position=np.array([obs.pose.x, obs.pose.y, obs.pose.z]),
        initial_yaw=obs.pose.yaw
    )
    adapter.set_observation(obs.rgb_image, obs.depth_map, np.array([obs.pose.x, obs.pose.y, obs.pose.z]), obs.pose.yaw)

    # Verify adapter serves real environment observation
    rgb, depth, intrinsics = adapter.get_camera_frame()
    assert rgb.shape == (360, 640, 3)
    assert depth.shape == (360, 640)
    assert intrinsics.width == 640 and intrinsics.height == 360

    # Command continuous forward motion along heading: v_forward = 4.5 m/s
    v_world = np.array([4.5 * math.cos(adapter.yaw), 4.5 * math.sin(adapter.yaw), 0.0])
    adapter.set_continuous_velocity(v_world[0], v_world[1], v_world[2], yaw_rate=0.0)
    action_code = adapter.quantize_action(adapter.vel, adapter.yaw_rate, dt=1.0)
    assert action_code == ACTION_MOVE_FORWARD_MED, f"Expected action 8 (MED), got {action_code}"

    # Step environment with quantized action
    next_obs, done, info = env.step(NavAction(action_code=action_code))
    adapter.set_observation(next_obs.rgb_image, next_obs.depth_map, np.array([next_obs.pose.x, next_obs.pose.y, next_obs.pose.z]), next_obs.pose.yaw)

    # Confirm telemetry reflects environment motion
    tel = adapter.get_telemetry()
    assert math.isclose(tel.position[0], next_obs.pose.x, abs_tol=1e-4)
    assert math.isclose(tel.position[1], next_obs.pose.y, abs_tol=1e-4)
