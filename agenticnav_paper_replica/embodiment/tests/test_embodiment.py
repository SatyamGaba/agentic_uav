"""Unit tests for embodiment base and mock embodiment."""

import math
import numpy as np
import pytest

from agenticnav_paper_replica.embodiment.base import StepObservation
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment


class TestMockGroundEmbodiment:
    def test_reset_and_observations(self):
        emb = MockGroundEmbodiment()
        obs = emb.reset({"goal_x": 0.0, "goal_z": -10.0})

        assert isinstance(obs, StepObservation)
        assert len(obs.rgb_views) == 4
        assert len(obs.depth_views) == 4
        assert obs.allocentric_pose == (0.0, 0.0, 0.0)
        assert obs.simulator_actions_taken == 0
        assert obs.is_done is False
        assert abs(emb.get_geodesic_distance_to_goal() - 10.0) < 1e-4

    def test_step_discrete_kinematics(self):
        emb = MockGroundEmbodiment()
        emb.reset({"goal_x": 0.0, "goal_z": -10.0})

        # Step forward 4 steps of 0.25m = 1.0m (along heading 0, which is -Z)
        obs = emb.step_discrete(turn_steps_15deg=0, forward_steps_025m=4)
        assert obs.simulator_actions_taken == 4
        assert len(obs.primitive_trajectory) == 4
        assert len(obs.primitive_geodesic_distances) == 4
        assert abs(obs.actual_distance_moved - 1.0) < 1e-4
        assert obs.collided is False
        x, z, yaw = obs.allocentric_pose
        assert abs(x) < 1e-4
        assert abs(z - (-1.0)) < 1e-4  # moved along -Z
        assert abs(yaw) < 1e-4
        # Distance to goal reduced from 10.0 to 9.0
        assert abs(emb.get_geodesic_distance_to_goal() - 9.0) < 1e-4

        # Turn left 6 steps of 15 deg = +90 deg (yaw = +pi/2)
        obs_turn = emb.step_discrete(turn_steps_15deg=6, forward_steps_025m=0)
        assert obs_turn.simulator_actions_taken == 10
        x, z, yaw = obs_turn.allocentric_pose
        assert abs(yaw - math.pi / 2.0) < 1e-4

        # Step forward 2 steps of 0.25m = 0.5m along heading +90 deg (along -X)
        obs_step = emb.step_discrete(turn_steps_15deg=0, forward_steps_025m=2)
        x, z, yaw = obs_step.allocentric_pose
        assert abs(x - (-0.5)) < 1e-4
        assert abs(z - (-1.0)) < 1e-4

