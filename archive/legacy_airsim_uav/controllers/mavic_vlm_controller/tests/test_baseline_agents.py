"""
Test suite for Baseline & Ablation Agents
=========================================
Verifies Phase 3 exit criteria:
- NaiveVLMVelocityAgent instantiates and emits Subgoal
- FrontierOnlyZeroShotAgent instantiates and emits Subgoal
- SceneGraphNoSoMAgent instantiates and emits Subgoal
- FullSystemAgent instantiates and emits Subgoal with end-to-end SoM + Scene Graph integration
"""

import asyncio
import numpy as np
import pytest

from contracts import (
    DroneTelemetry,
    CameraIntrinsics,
    SubgoalIntent,
    Subgoal,
    MarkSelection,
)
from cognitive.vlm_client import MockVLMClient, SelfConsistentVLMWrapper
from cognitive.som_grounder import MockDetector
from cognitive.intent_arbiter import TaskDecompositionResult, DecomposedStage
from cognitive.baseline_agents import (
    NaiveVLMVelocityAgent,
    FrontierOnlyZeroShotAgent,
    SceneGraphNoSoMAgent,
    FullSystemAgent,
)


@pytest.fixture
def mock_telemetry():
    return DroneTelemetry(
        position=np.array([5.0, 10.0, 6.0]),
        velocity=np.array([1.5, 0.0, 0.0]),
        yaw=0.0,
        timestamp=25.0,
    )


@pytest.fixture
def mock_intrinsics():
    K = np.array([
        [200.0, 0.0, 200.0],
        [0.0, 200.0, 150.0],
        [0.0, 0.0, 1.0],
    ])
    return CameraIntrinsics(
        K=K,
        width=400,
        height=300,
        vfov=0.8,
        hfov=1.0,
    )


def test_naive_vlm_velocity_agent(mock_telemetry):
    async def _run():
        mock_subgoal = Subgoal(
            subgoal_id=1,
            intent=SubgoalIntent.CORRIDOR_TRANSIT,
            waypoint_xyz=(15.0, 10.0, 6.0),
            v_target=3.0,
        )
        mock_client = MockVLMClient([mock_subgoal])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        agent = NaiveVLMVelocityAgent(wrapper)

        rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        depth = np.full((300, 400), 10.0, dtype=np.float32)

        res = await agent.step(rgb, depth, mock_telemetry, "Fly east down the road")
        assert isinstance(res, Subgoal)
        assert res.intent == SubgoalIntent.CORRIDOR_TRANSIT
        assert res.waypoint_xyz == (15.0, 10.0, 6.0)

    asyncio.run(_run())


def test_frontier_only_agent(mock_telemetry):
    async def _run():
        mock_client = MockVLMClient([])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        agent = FrontierOnlyZeroShotAgent(wrapper, default_altitude=8.0)

        rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        depth = np.full((300, 400), 10.0, dtype=np.float32)

        res = await agent.step(rgb, depth, mock_telemetry, "Explore the unknown area")
        assert isinstance(res, Subgoal)
        assert res.intent == SubgoalIntent.SEARCH_AND_ORIENT
        assert res.waypoint_xyz[2] == 8.0  # Default altitude

    asyncio.run(_run())


def test_scene_graph_no_som_agent(mock_telemetry, mock_intrinsics):
    async def _run():
        mock_subgoal = Subgoal(
            subgoal_id=1,
            intent=SubgoalIntent.TARGET_APPROACH,
            waypoint_xyz=(20.0, 10.0, 6.0),
            v_target=2.0,
        )
        mock_client = MockVLMClient([mock_subgoal])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        agent = SceneGraphNoSoMAgent(wrapper, mock_intrinsics)

        rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        depth = np.full((300, 400), 10.0, dtype=np.float32)

        res = await agent.step(rgb, depth, mock_telemetry, "Approach the red building")
        assert isinstance(res, Subgoal)
        assert res.intent == SubgoalIntent.TARGET_APPROACH

    asyncio.run(_run())


def test_full_system_agent_end_to_end(mock_telemetry, mock_intrinsics):
    async def _run():
        # Setup mock VLM responses:
        # 1. Task decomposition
        mock_decomp = TaskDecompositionResult(
            stages=[
                DecomposedStage(
                    stage_id=1,
                    text="Locate and approach the manor house.",
                    detector_queries=["manor house", "chimney"],
                )
            ]
        )
        # 2. Set-of-Mark selection
        mock_selection = MarkSelection(
            selected_mark_id=1,
            confidence=0.91,
            reasoning="Mark #1 clearly identifies the manor house.",
        )
        # 3. Subgoal arbitration
        mock_subgoal = Subgoal(
            subgoal_id=1,
            intent=SubgoalIntent.TARGET_APPROACH,
            waypoint_xyz=(18.0, 10.0, 6.0),
            v_target=2.5,
        )

        mock_client = MockVLMClient([mock_decomp, mock_selection, mock_subgoal])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)

        # Mock detector proposes candidate #1 at center of image at 10m depth
        detector = MockDetector([
            ((0.45, 0.45, 0.55, 0.55), "manor house", 0.95)
        ])

        agent = FullSystemAgent(wrapper, detector, mock_intrinsics)

        rgb = np.zeros((300, 400, 3), dtype=np.uint8)
        depth = np.full((300, 400), 10.0, dtype=np.float32)

        subgoal = await agent.step(rgb, depth, mock_telemetry, "Find manor and approach.")
        assert isinstance(subgoal, Subgoal)
        assert subgoal.intent == SubgoalIntent.TARGET_APPROACH
        assert subgoal.target_node_id == 1  # Successfully grounded to SceneNode #1!
        assert len(agent.scene_graph.nodes) == 1

    asyncio.run(_run())
