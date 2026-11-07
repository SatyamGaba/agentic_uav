"""End-to-end integration tests for AgenticNav execution runner and privilege firewall."""

from pathlib import Path
import numpy as np
import pytest

from agenticnav_paper_replica.configs.default_config import ReplicaConfig
from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
from agenticnav_paper_replica.harness.runner import run_episode, EpisodeRunResult
from agenticnav_paper_replica.traces.trace_logger import TraceLogger


class TestEndToEndRunner:
    def test_successful_navigation_episode(self, tmp_path):
        config = ReplicaConfig()
        embodiment = MockGroundEmbodiment()
        logger = TraceLogger(tmp_path / "traces")

        episode_data = {
            "episode_id": "test_ep_001",
            "scene_id": "test_scene",
            "instruction": "Walk straight ahead down the hallway to the counter.",
            "goal_x": 0.0,
            "goal_z": -2.0,  # Goal is 2.0m ahead
        }

        # Mock VLM: Step 1 queries depth, then moves forward to (u=0.5, v=0.5)
        # Step 2 calls stop
        turn_counter = 0

        def mock_vlm(messages, tools):
            nonlocal turn_counter
            turn_counter += 1

            # Privilege firewall check: messages must NEVER contain goal_x or goal_z
            prompt_str = str(messages)
            assert "goal_x" not in prompt_str
            assert "goal_z" not in prompt_str
            assert "PathFinder" not in prompt_str

            if turn_counter == 1:
                # Query depth
                return {
                    "content": "",
                    "tool_calls": [{"name": "query_depth", "arguments": {"points": [{"view": 0, "u": 0.5, "v": 0.5}]}}],
                }
            elif turn_counter == 2:
                # Move forward
                return {
                    "content": "",
                    "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Moving toward target"}}],
                }
            else:
                # Stop
                return {
                    "content": "",
                    "tool_calls": [{"name": "stop", "arguments": {"reason": "Arrived at counter"}}],
                }

        res: EpisodeRunResult = run_episode(
            episode_data=episode_data,
            embodiment=embodiment,
            config=config,
            vlm_client_fn=mock_vlm,
            trace_logger=logger,
        )

        assert res.metrics.success is True
        assert res.metrics.spl > 0.0
        assert res.failure_category is None
        assert res.terminated_deadlock is False
        assert logger.jsonl_path.exists()

    def test_deadlock_episode(self, tmp_path):
        config = ReplicaConfig(max_consecutive_reselects=3)
        embodiment = MockGroundEmbodiment()
        logger = TraceLogger(tmp_path / "traces")

        episode_data = {
            "episode_id": "test_ep_deadlock",
            "scene_id": "test_scene",
            "instruction": "Navigate forward.",
            "goal_x": 0.0,
            "goal_z": -10.0,
        }

        # VLM repeatedly calls a point that yields zero displacement (v=0.99 with high floor)
        # causing zero_displacement reselect
        def mock_blocked_vlm(messages, tools):
            # Target with zero distance
            return {
                "content": "",
                "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Try moving"}}],
            }

        # Mock the embodiment depth views to return 0.35m everywhere (< 0.25m after 0.30m margin)
        # To do this, let's subclass MockGroundEmbodiment
        class BlockedEmbodiment(MockGroundEmbodiment):
            def _make_obs(self):
                obs = super()._make_obs()
                obs.depth_views = {v: np.full((480, 640), 0.35, dtype=np.float32) for v in range(4)}
                return obs

        blocked_emb = BlockedEmbodiment()
        res = run_episode(
            episode_data=episode_data,
            embodiment=blocked_emb,
            config=config,
            vlm_client_fn=mock_blocked_vlm,
            trace_logger=logger,
        )

        assert res.terminated_deadlock is True
        assert res.failure_category == "safety_deadlock"
        assert res.metrics.success is False

