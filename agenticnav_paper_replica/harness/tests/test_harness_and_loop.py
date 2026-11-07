"""Unit tests for tool schemas, executor, and multi-turn step loop (v7 Specification)."""

import json
import numpy as np
import pytest

from agenticnav_paper_replica.geometry.pinhole import CameraConfig
from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker
from agenticnav_paper_replica.harness.tools import get_available_tools
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.harness.step_loop import StepDecisionLoop, StepLoopResult
from agenticnav_paper_replica.memory.history import ActionHistory
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache


class TestToolSchemas:
    def test_full_schemas(self):
        tools = get_available_tools(enable_depth=True, enable_recall=True)
        names = [t["function"]["name"] for t in tools]
        assert "query_depth" in names
        assert "recall" in names
        assert "move_to" in names
        assert "stop" in names

    def test_ablation_stripping(self):
        tools_no_depth = get_available_tools(enable_depth=False, enable_recall=True)
        names = [t["function"]["name"] for t in tools_no_depth]
        assert "query_depth" not in names
        assert "recall" in names

        tools_no_recall = get_available_tools(enable_depth=True, enable_recall=False)
        names = [t["function"]["name"] for t in tools_no_recall]
        assert "recall" not in names
        assert "query_depth" in names


class TestToolExecutor:
    def setup_method(self):
        self.config = CameraConfig()
        self.checker = GroundCorridorChecker(robot_radius=0.18, base_margin=0.30)
        self.image_cache = EpisodicImageCache()
        self.executor = ToolExecutor(self.config, self.checker, self.image_cache)
        self.depth_views = {v: np.full((480, 640), 2.0, dtype=np.float32) for v in range(4)}

    def test_query_depth_valid_and_invalid(self):
        pts = [{"view": 0, "u": 0.5, "v": 0.5}]
        res = self.executor.execute_query_depth(pts, self.depth_views)
        q_res = res["query_results"][0]
        assert "error" not in q_res
        assert abs(q_res["depth_m"] - 2.0) < 1e-2
        assert abs(q_res["min_m"] - 2.0) < 1e-2

        # Invalid depth (NaN, Inf, or saturated 1.0)
        self.depth_views[0][240, 320] = 0.0
        res_inv = self.executor.execute_query_depth(pts, self.depth_views)
        assert res_inv["query_results"][0]["error"] == "invalid_depth"

    def test_recall_valid_and_invalid(self):
        self.image_cache.store_node_observations(node=1, rgb_views={0: np.zeros((10, 10, 3), dtype=np.uint8)})
        self.image_cache.set_current_node(node=2)

        res_ok = self.executor.execute_recall(node=1, view=0)
        assert res_ok["status"] == "success"

        res_bad = self.executor.execute_recall(node=2, view=0)
        assert res_bad["status"] == "error"

    def test_query_depth_accepts_json_string_points(self):
        points = json.dumps([{"view": 0, "u": 0.5, "v": 0.5}])
        res = self.executor.execute_query_depth(points, self.depth_views)
        assert res["query_results"][0]["depth_m"] == pytest.approx(2.0, abs=1e-2)

    def test_move_to_executed_and_reselect(self):
        obstacles = np.empty((0, 2))
        res = self.executor.execute_move_to(
            view=0, u=0.5, v=0.5, reason="Move straight ahead",
            depth_views=self.depth_views, obstacle_points_agent_xz=obstacles,
        )
        assert res["status"] == "executed"
        assert res["rho_m"] == 1.50
        assert res["forward_steps"] == 6

        # Obstacle in front at (0, -1.0)
        obs_blocked = np.array([[0.0, -1.0]])
        res_blocked = self.executor.execute_move_to(
            view=0, u=0.5, v=0.5, reason="Move straight ahead",
            depth_views=self.depth_views, obstacle_points_agent_xz=obs_blocked,
        )
        assert res_blocked["status"] == "reselect"
        assert res_blocked["cause"] == "obstacle"


class TestStepDecisionLoop:
    def setup_method(self):
        self.config = CameraConfig()
        self.checker = GroundCorridorChecker()
        self.image_cache = EpisodicImageCache()
        self.executor = ToolExecutor(self.config, self.checker, self.image_cache)
        self.loop = StepDecisionLoop(
            executor=self.executor,
            max_evidence_queries=4,
            max_consecutive_reselects=5,
            max_malformed_calls=3,
        )
        self.depth_views = {v: np.full((480, 640), 2.5, dtype=np.float32) for v in range(4)}
        self.rgb_views = {v: np.full((480, 640, 3), 128, dtype=np.uint8) for v in range(4)}
        self.history = ActionHistory()

    def test_evidence_then_move_flow(self):
        calls = [
            {"content": "", "tool_calls": [{"name": "query_depth", "arguments": {"points": [{"view": 0, "u": 0.5, "v": 0.5}]}}]},
            {"content": "", "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Path is clear"}}]},
        ]
        call_idx = 0

        def mock_vlm(messages, tools):
            nonlocal call_idx
            res = calls[call_idx]
            call_idx += 1
            return res

        result: StepLoopResult = self.loop.run_step(
            instruction="Walk down the corridor",
            current_node=1,
            history=self.history,
            rgb_views=self.rgb_views,
            depth_views=self.depth_views,
            obstacle_points_agent_xz=np.empty((0, 2)),
            bev_image=None,
            vlm_client_fn=mock_vlm,
        )

        assert result.action_type == "move_to"
        assert result.evidence_query_count == 1
        assert result.action_payload["status"] == "executed"

    def test_deferred_response_when_evidence_and_move_called_together(self):
        # Model returns query_depth AND move_to in the same turn
        # Turn 1: query_depth executed, move_to receives deferred result.
        # Turn 2: Model reissues move_to alone.
        turn_idx = 0

        def mock_vlm(messages, tools):
            nonlocal turn_idx
            turn_idx += 1
            if turn_idx == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {"name": "query_depth", "arguments": {"points": [{"view": 0, "u": 0.5, "v": 0.5}]}},
                        {"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "First guess"}},
                    ],
                }
            else:
                # Check that deferred result was provided
                assert any("deferred" in str(m) for m in messages)
                return {
                    "content": "",
                    "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Confirmed move"}}],
                }

        result = self.loop.run_step(
            instruction="Walk forward",
            current_node=1,
            history=self.history,
            rgb_views=self.rgb_views,
            depth_views=self.depth_views,
            obstacle_points_agent_xz=np.empty((0, 2)),
            bev_image=None,
            vlm_client_fn=mock_vlm,
        )

        assert result.action_type == "move_to"
        assert result.action_payload["status"] == "executed"

    def test_reselect_deadlock_trigger(self):
        obs = np.array([[0.0, -1.0]])

        def mock_blocked_vlm(messages, tools):
            return {"content": "", "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Try blocked path"}}]}

        result: StepLoopResult = self.loop.run_step(
            instruction="Walk forward",
            current_node=1,
            history=self.history,
            rgb_views=self.rgb_views,
            depth_views=self.depth_views,
            obstacle_points_agent_xz=obs,
            bev_image=None,
            vlm_client_fn=mock_blocked_vlm,
        )

        assert result.action_type == "deadlock"
        assert result.terminated_deadlock is True
        assert result.consecutive_reselect_count == 5

    def test_duplicate_target_triggers_arrival_verification(self):
        obs = np.array([[0.0, -1.0]])

        def mock_duplicate_vlm(messages, tools):
            return {"content": "", "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Repeat blocked target"}}]}

        result: StepLoopResult = self.loop.run_step(
            instruction="Walk forward",
            current_node=1,
            history=self.history,
            rgb_views=self.rgb_views,
            depth_views=self.depth_views,
            obstacle_points_agent_xz=obs,
            bev_image=None,
            vlm_client_fn=mock_duplicate_vlm,
        )

        assert result.arrival_verification_triggered is True
        assert any(turn.get("arrival_verification") for turn in result.turn_log)
        assert result.action_type == "deadlock"

    def test_arrival_verification_turn_can_stop(self):
        obs = np.array([[0.0, -1.0]])
        turn = 0

        def mock_verify_then_stop(messages, tools):
            nonlocal turn
            turn += 1
            if turn <= 2:
                return {"content": "", "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Blocked"}}]}
            assert any(isinstance(m, dict) and m.get("arrival_verification") for m in messages)
            return {"content": "", "tool_calls": [{"name": "stop", "arguments": {"reason": "Destination landmark visible"}}]}

        result: StepLoopResult = self.loop.run_step(
            instruction="Stop at the door",
            current_node=1,
            history=self.history,
            rgb_views=self.rgb_views,
            depth_views=self.depth_views,
            obstacle_points_agent_xz=obs,
            bev_image=None,
            vlm_client_fn=mock_verify_then_stop,
        )

        assert result.action_type == "stop"
        assert result.arrival_verification_triggered is True

    def test_protocol_violation_trigger(self):
        # Model returns empty messages (no tool calls) 3 times -> triggers protocol violation
        def mock_empty_vlm(messages, tools):
            return {"content": "I am thinking but calling no tool.", "tool_calls": []}

        result: StepLoopResult = self.loop.run_step(
            instruction="Walk forward",
            current_node=1,
            history=self.history,
            rgb_views=self.rgb_views,
            depth_views=self.depth_views,
            obstacle_points_agent_xz=np.empty((0, 2)),
            bev_image=None,
            vlm_client_fn=mock_empty_vlm,
        )

        assert result.action_type == "protocol_violation"
        assert result.terminated_protocol_violation is True
        assert result.malformed_call_count == 3
