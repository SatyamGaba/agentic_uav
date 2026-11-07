"""
Test suite for Intent Arbiter & Mission Decomposer
==================================================
Tests Phase 1 exit criteria:
- Non-linear DAG decomposition on conditional instructions
- Backtracking support
- Cyclical patrol support
- Subgoal arbitration with mocked VLM client
"""

import asyncio
import math
import numpy as np
import pytest
from contracts import (
    DroneTelemetry,
    SubgoalIntent,
    Subgoal,
)
from cognitive.vlm_client import MockVLMClient, SelfConsistentVLMWrapper
from cognitive.intent_arbiter import (
    IntentArbiter,
    DecomposedStage,
    TaskDecompositionResult,
)


def test_conditional_branching_decomposition():
    """Verifies that a conditional instruction creates alternative branch targets."""
    async def _run():
        mock_decomp = TaskDecompositionResult(
            stages=[
                DecomposedStage(
                    stage_id=1,
                    text="Fly along Main Street towards the north bridge.",
                    detector_queries=["bridge", "river", "road"],
                    condition="if north bridge is blocked",
                    fallback_stage_id=3,
                ),
                DecomposedStage(
                    stage_id=2,
                    text="Cross the north bridge and land at clock tower plaza.",
                    detector_queries=["clock tower", "plaza"],
                ),
                DecomposedStage(
                    stage_id=3,
                    text="Divert east along riverbank to pedestrian crossing.",
                    detector_queries=["pedestrian crossing", "footbridge"],
                ),
            ],
            is_cyclical=False,
        )

        mock_client = MockVLMClient([mock_decomp])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        arbiter = IntentArbiter(wrapper)

        res = await arbiter.decompose_mission(
            "Fly along Main Street. If the north bridge is blocked, divert east to the pedestrian crossing, otherwise proceed to clock tower."
        )
        assert len(res.stages) == 3
        assert arbiter.get_current_stage().stage_id == 1

        # Simulate condition trigger: bridge is blocked -> branch to stage 3
        branched = arbiter.branch_to_stage(res.stages[0].fallback_stage_id)
        assert branched is True
        assert arbiter.get_current_stage().stage_id == 3
        assert "Divert east" in arbiter.get_current_stage().text

    asyncio.run(_run())


def test_backtracking_decomposition():
    """Verifies that an instruction requiring return to origin is decomposed as a distinct stage."""
    async def _run():
        mock_decomp = TaskDecompositionResult(
            stages=[
                DecomposedStage(
                    stage_id=1,
                    text="Climb to 8m and transit to Building A rooftop.",
                    detector_queries=["Building A", "rooftop", "solar panel"],
                ),
                DecomposedStage(
                    stage_id=2,
                    text="Perform proximal visual survey of solar panel array.",
                    detector_queries=["solar panel", "inverter box"],
                ),
                DecomposedStage(
                    stage_id=3,
                    text="Backtrack directly to starting courtyard at origin.",
                    detector_queries=["courtyard", "fountain"],
                ),
                DecomposedStage(
                    stage_id=4,
                    text="Navigate south-west to final helipad destination.",
                    detector_queries=["helipad", "landing H"],
                ),
            ],
            is_cyclical=False,
        )

        mock_client = MockVLMClient([mock_decomp])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        arbiter = IntentArbiter(wrapper)

        res = await arbiter.decompose_mission(
            "Inspect solar panels on Building A, then return to starting courtyard before proceeding to helipad."
        )
        assert len(res.stages) == 4
        # Sequential progression
        assert arbiter.advance_stage() is True  # Stage 2
        assert arbiter.advance_stage() is True  # Stage 3 (Backtrack)
        assert arbiter.get_current_stage().stage_id == 3
        assert "Backtrack" in arbiter.get_current_stage().text

    asyncio.run(_run())


def test_cyclical_patrol_loop():
    """Verifies that a continuous patrol instruction loops back to stage 0 after completing."""
    async def _run():
        mock_decomp = TaskDecompositionResult(
            stages=[
                DecomposedStage(stage_id=1, text="Inspect North gate", detector_queries=["north gate"]),
                DecomposedStage(stage_id=2, text="Inspect East perimeter", detector_queries=["east wall"]),
                DecomposedStage(stage_id=3, text="Inspect South gate", detector_queries=["south gate"]),
                DecomposedStage(stage_id=4, text="Inspect West perimeter", detector_queries=["west wall"]),
            ],
            is_cyclical=True,
        )

        mock_client = MockVLMClient([mock_decomp])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        arbiter = IntentArbiter(wrapper)

        await arbiter.decompose_mission("Continuously patrol the four perimeter sectors of the compound.")
        assert arbiter.get_current_stage().stage_id == 1
        arbiter.advance_stage()  # -> 2
        arbiter.advance_stage()  # -> 3
        arbiter.advance_stage()  # -> 4
        assert arbiter.get_current_stage().stage_id == 4

        # Crucial test: advancement from last stage loops back to Stage 1 because is_cyclical=True
        cycled = arbiter.advance_stage()
        assert cycled is True
        assert arbiter.get_current_stage().stage_id == 1

    asyncio.run(_run())


def test_subgoal_arbitration_call():
    """Verifies that arbitrate produces a valid typed Subgoal from telemetry and scene graph."""
    async def _run():
        mock_subgoal = Subgoal(
            subgoal_id=99,
            intent=SubgoalIntent.CORRIDOR_TRANSIT,
            waypoint_xyz=(20.0, 35.0, 6.0),
            v_target=3.0,
            caution_level="medium",
            termination_condition="reached_street_intersection",
        )

        mock_client = MockVLMClient([mock_subgoal])
        wrapper = SelfConsistentVLMWrapper(mock_client, n_samples=1)
        arbiter = IntentArbiter(wrapper)

        telemetry = DroneTelemetry(
            position=np.array([0.0, 0.0, 5.0]),
            velocity=np.array([1.0, 0.0, 0.0]),
            yaw=0.0,
            timestamp=10.0,
        )

        subgoal = await arbiter.arbitrate(
            instruction_raw="Transit along the central avenue.",
            rgb_image=None,
            scene_graph_text="[Scene Graph: 2 confirmed buildings]",
            telemetry=telemetry,
            history=[],
        )

        assert subgoal.subgoal_id == 1  # Arbiter assigned sequence counter
        assert subgoal.intent == SubgoalIntent.CORRIDOR_TRANSIT
        assert subgoal.waypoint_xyz == (20.0, 35.0, 6.0)
        assert subgoal.issued_at == 10.0

    asyncio.run(_run())
