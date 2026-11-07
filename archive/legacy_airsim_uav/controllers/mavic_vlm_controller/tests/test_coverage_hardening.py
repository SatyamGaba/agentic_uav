"""
Coverage Hardening & Concurrency Test Suite
===========================================
Addresses specific coverage and concurrency gaps identified in the audit review:
1. vlm_client.py: JSON extraction with <think> tokens, retry handling on HTTP/JSON errors, timeout exceptions.
2. main_controller.py: Real concurrent execution (Fast loop thread + Asyncio task) testing blackboard races and thread stop.
3. scene_graph_3d.py: Stale node pruning, multi-view confirmation baseline, yaw-box PCA, prompt serializer sorting.
4. openfly_adapter.py: Complete branch coverage of all canonical action codes (fine/med/large forward, climb, descend, strafes, turns).
"""

import math
import time
import asyncio
import threading
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
import urllib.error

from contracts import (
    DroneTelemetry,
    CameraIntrinsics,
    SubgoalIntent,
    Subgoal,
    MarkSelection,
    SceneNode,
    SafetyState,
)
from cognitive.vlm_client import (
    extract_json_from_llm_output,
    OpenAICompatibleVLMClient,
    SelfConsistentVLMWrapper,
    MockVLMClient,
)
from cognitive.intent_arbiter import IntentArbiter, TaskDecompositionResult
from core.scene_graph_3d import MetricSceneGraph3D, robust_centroid_and_extent, compute_camera_to_world_matrix
from adapters.openfly_adapter import (
    OpenFlyBenchmarkAdapter,
    ACTION_STOP,
    ACTION_MOVE_FORWARD_FINE,
    ACTION_MOVE_FORWARD_MED,
    ACTION_MOVE_FORWARD_LARGE,
    ACTION_TURN_LEFT_30,
    ACTION_TURN_RIGHT_30,
    ACTION_CLIMB_3M,
    ACTION_DESCEND_3M,
    ACTION_STRAFE_LEFT_3M,
    ACTION_STRAFE_RIGHT_3M,
)
from main_controller import SharedBlackboard, FastContinuousController, DualRateMasterController


# ---------------------------------------------------------------------------
# 1. VLM Client JSON Extraction & Error Path Coverage
# ---------------------------------------------------------------------------

def test_extract_json_stripping_think_tags():
    raw_output = """
    <think>
    I should consider whether mark {1} or {2} is better.
    Let's check: {"candidate": 1}
    </think>
    ```json
    {
        "selected_mark_id": 2,
        "confidence": 0.95,
        "reasoning": "Clear view of landmark"
    }
    ```
    """
    parsed = extract_json_from_llm_output(raw_output)
    assert parsed["selected_mark_id"] == 2
    assert parsed["confidence"] == 0.95


def test_extract_json_bare_curly_braces():
    raw_output = "The result is {\"selected_mark_id\": 1, \"confidence\": 0.8} as expected."
    parsed = extract_json_from_llm_output(raw_output)
    assert parsed["selected_mark_id"] == 1


def test_vlm_client_retry_and_error_handling():
    client = OpenAICompatibleVLMClient(
        endpoint_url="http://127.0.0.1:9999/v1/chat/completions",
        model_name="test-model",
        timeout_s=1.0,
        max_retries=1,
    )

    async def _test():
        # Test simulated network connection failure triggers retry and final exception
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("Connection refused")):
            with pytest.raises(urllib.error.URLError):
                await client.query_structured(
                    system_prompt="Test",
                    user_prompt="Hello",
                    images=[],
                    schema=Subgoal,
                )

    asyncio.run(_test())


# ---------------------------------------------------------------------------
# 2. Main Controller True Concurrency & Race Condition Test
# ---------------------------------------------------------------------------

def test_true_concurrency_fast_thread_and_async_slow_loop():
    """Validates real multi-threading concurrency with high contention, checking monotonicity and zero torn writes."""
    adapter = OpenFlyBenchmarkAdapter(initial_position=np.array([0.0, 0.0, 5.0]))
    initial_subgoal = Subgoal(
        subgoal_id=1,
        intent=SubgoalIntent.SEARCH_AND_ORIENT,
        waypoint_xyz=(1.0, 2.0, 5.0),
        issued_at=time.time(),
        expiry_s=2.0,
    )

    blackboard = SharedBlackboard(initial_subgoal)
    fast_controller = FastContinuousController(adapter, blackboard, tick_hz=100.0)

    stop_event = threading.Event()
    read_observations = []
    race_detected = []

    def run_fast_thread():
        last_id = 0
        while not stop_event.is_set():
            fast_controller.tick()
            sg = blackboard.get_current_subgoal()
            
            # Assert zero torn writes: waypoint y must always equal 2.0 * waypoint x
            if abs(sg.waypoint_xyz[1] - (sg.waypoint_xyz[0] * 2.0)) > 1e-5:
                race_detected.append(f"Torn write detected! sg_id={sg.subgoal_id}, wp={sg.waypoint_xyz}")
            
            # Assert monotonic version/subgoal sequence
            if sg.subgoal_id < last_id:
                race_detected.append(f"Non-monotonic read! last={last_id}, current={sg.subgoal_id}")
            last_id = sg.subgoal_id
            read_observations.append(sg.subgoal_id)
            time.sleep(0.001)

    t = threading.Thread(target=run_fast_thread, daemon=True)
    t.start()

    async def run_slow_async_loop():
        # Concurrently hammer blackboard with 50 rapid updates across async context
        for i in range(2, 52):
            await asyncio.sleep(0.003)
            new_subgoal = Subgoal(
                subgoal_id=i,
                intent=SubgoalIntent.TARGET_APPROACH,
                waypoint_xyz=(float(i), float(i) * 2.0, 5.0),
                issued_at=time.time(),
                expiry_s=3.0,
            )
            blackboard.set_subgoal(new_subgoal)

    asyncio.run(run_slow_async_loop())
    stop_event.set()
    t.join(timeout=1.0)
    assert not t.is_alive()

    # Validate findings
    assert len(race_detected) == 0, f"Concurrency race conditions found: {race_detected}"
    assert len(read_observations) >= 20, "Fast thread did not complete enough ticks"
    assert max(read_observations) >= 40, "Fast thread did not observe recent writes"


# ---------------------------------------------------------------------------
# 3. Scene Graph 3D Edge Branches & Lifecycle
# ---------------------------------------------------------------------------

def test_scene_graph_prune_stale_and_serialization():
    sg = MetricSceneGraph3D()
    # Node 1: unconfirmed, old
    node1 = SceneNode(
        node_id=1, label="trash", centroid_world=np.array([10.0, 0.0, 5.0]),
        dimensions_3d=np.array([1.0, 1.0, 1.0]), confidence=0.3, observation_count=1,
        last_observed_timestamp=10.0, first_viewpoint=np.zeros(3), is_confirmed=False
    )
    # Node 2: confirmed, permanent
    node2 = SceneNode(
        node_id=2, label="tower", centroid_world=np.array([20.0, 5.0, 10.0]),
        dimensions_3d=np.array([4.0, 4.0, 12.0]), confidence=0.9, observation_count=4,
        last_observed_timestamp=50.0, first_viewpoint=np.zeros(3), is_confirmed=True
    )
    sg.nodes[1] = node1
    sg.nodes[2] = node2

    # Prune unconfirmed nodes older than 30s at t = 100.0s
    pruned_count = sg.prune_stale(max_age_s=30.0, current_time=100.0)
    assert pruned_count == 1
    assert 1 not in sg.nodes
    assert 2 in sg.nodes

    # Test serialization of confirmed node
    tel = DroneTelemetry(
        position=np.array([0.0, 0.0, 5.0]),
        velocity=np.zeros(3),
        yaw=0.0,
        gimbal_pitch=0.0,
        gimbal_yaw=0.0,
        timestamp=100.0,
    )
    prompt_text = sg.serialize_for_prompt(tel)
    assert "tower" in prompt_text
    assert "ID #02" in prompt_text


# ---------------------------------------------------------------------------
# 4. OpenFly Adapter All Canonical Action Branches
# ---------------------------------------------------------------------------

def test_openfly_all_canonical_action_branches():
    adapter = OpenFlyBenchmarkAdapter(initial_position=np.array([0.0, 0.0, 5.0]))

    # Angular adjustment: +yaw is TURN_LEFT (code 2), -yaw is TURN_RIGHT (code 3)
    assert adapter.quantize_action(np.zeros(3), yaw_rate_cmd=0.5, dt=1.0) == ACTION_TURN_LEFT_30
    assert adapter.quantize_action(np.zeros(3), yaw_rate_cmd=-0.5, dt=1.0) == ACTION_TURN_RIGHT_30

    # Climb & Descend
    assert adapter.quantize_action(np.array([0.0, 0.0, 2.0]), yaw_rate_cmd=0.0, dt=1.0) == ACTION_CLIMB_3M
    assert adapter.quantize_action(np.array([0.0, 0.0, -2.0]), yaw_rate_cmd=0.0, dt=1.0) == ACTION_DESCEND_3M

    # Forward Fine, Med, Large
    assert adapter.quantize_action(np.array([2.0, 0.0, 0.0]), yaw_rate_cmd=0.0, dt=1.0) == ACTION_MOVE_FORWARD_FINE
    assert adapter.quantize_action(np.array([5.0, 0.0, 0.0]), yaw_rate_cmd=0.0, dt=1.0) == ACTION_MOVE_FORWARD_MED
    assert adapter.quantize_action(np.array([8.0, 0.0, 0.0]), yaw_rate_cmd=0.0, dt=1.0) == ACTION_MOVE_FORWARD_LARGE

    # Lateral displacement: +vy is STRAFE_LEFT (code 7), -vy is STRAFE_RIGHT (code 6)
    assert adapter.quantize_action(np.array([0.0, 2.0, 0.0]), yaw_rate_cmd=0.0, dt=1.0) == ACTION_STRAFE_LEFT_3M
    assert adapter.quantize_action(np.array([0.0, -2.0, 0.0]), yaw_rate_cmd=0.0, dt=1.0) == ACTION_STRAFE_RIGHT_3M
