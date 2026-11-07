"""
Intent Arbiter & Task Decomposer
================================
Translates natural language mission instructions into dynamic, DAG-friendly
sub-instruction stages and arbitrates 3D subgoals for System 1 without rigid FSMs.
"""

from __future__ import annotations
import math
import logging
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, Field

from contracts import (
    DroneTelemetry,
    SubgoalIntent,
    Subgoal,
)
from cognitive.vlm_client import SelfConsistentVLMWrapper

logger = logging.getLogger("MavicVLM")


class DecomposedStage(BaseModel):
    """A semantic sub-instruction stage with associated detector queries."""
    stage_id: int
    text: str
    detector_queries: List[str] = Field(default_factory=list)
    condition: Optional[str] = None          # Branch condition (e.g. "if path blocked")
    fallback_stage_id: Optional[int] = None  # Jump target if condition fails


class TaskDecompositionResult(BaseModel):
    """The structured output of mission instruction decomposition."""
    stages: List[DecomposedStage] = Field(default_factory=list)
    is_cyclical: bool = False                # E.g. continuous perimeter patrol


class ArbitrationMode(str, Enum):
    STRUCTURED_SCHEMA = "structured_schema"
    PROGRAM_SYNTHESIS = "program_synthesis"


class IntentArbiter:
    """Deliberative cognitive manager that decomposes instructions and arbitrates subgoals."""

    DECOMPOSITION_SYSTEM = (
        "You are an expert autonomous UAV mission planner. Given an aerial mission instruction, "
        "decompose it into a sequence of semantic flight stages.\n"
        "RULES:\n"
        "1. Do NOT assume a linear path if the prompt mentions conditionals ('if', 'unless', 'or') or loops ('patrol', 'circle').\n"
        "2. For each stage, output:\n"
        "   - 'text': Clear concise instruction for that stage.\n"
        "   - 'detector_queries': 2 to 4 visual search phrases (e.g. ['red brick house', 'chimney', 'roof']).\n"
        "   - 'condition': Optional string describing when to branch or skip.\n"
        "   - 'fallback_stage_id': Stage ID to jump to if target missing or condition triggered.\n"
        "Output ONLY a valid JSON conforming to TaskDecompositionResult."
    )

    ARBITRATION_SYSTEM = (
        "You are the high-level cognitive arbitrator for an autonomous continuous 6-DoF UAV.\n"
        "You are provided with:\n"
        "1. Active Mission Stage & History\n"
        "2. Current 3D Metric Scene Graph (grounded confirmed objects)\n"
        "3. UAV Current Telemetry (Position, Altitude, Heading, Speed)\n"
        "Your task: Emit the next Subgoal directive for the low-level continuous controller.\n"
        "RULES:\n"
        "- Choose 'intent' from: INTENT_SEARCH_AND_ORIENT, INTENT_CORRIDOR_TRANSIT, INTENT_TARGET_APPROACH, INTENT_SURVEY_INSPECT, INTENT_TERMINAL_VERIFY.\n"
        "- 'waypoint_xyz': Specify 3D target coordinates in World ENU [X, Y, Z]. If approaching a scene graph object, use its confirmed coordinates.\n"
        "- 'caution_level': 'high' when within 8m of obstacles or in narrow spaces; 'low' for open high-altitude transit.\n"
        "- 'termination_condition': Precise event when this subgoal is fulfilled (e.g. 'arrival_within_3m', 'landmark_in_fov').\n"
        "Output ONLY valid JSON conforming to Subgoal schema."
    )

    def __init__(
        self,
        vlm_wrapper: SelfConsistentVLMWrapper,
        mode: ArbitrationMode = ArbitrationMode.STRUCTURED_SCHEMA,
    ):
        self.vlm = vlm_wrapper
        self.mode = mode
        self.current_decomposition: Optional[TaskDecompositionResult] = None
        self.current_stage_idx: int = 0
        self.subgoal_counter: int = 1

    async def decompose_mission(self, instruction: str) -> TaskDecompositionResult:
        """Decomposes an open-ended natural language instruction into DAG stages."""
        user_prompt = f"Decompose this mission instruction:\n\"{instruction}\""
        res = await self.vlm.query_with_voting(
            system_prompt=self.DECOMPOSITION_SYSTEM,
            user_prompt=user_prompt,
            images=[],
            schema=TaskDecompositionResult,
        )
        self.current_decomposition = res
        self.current_stage_idx = 0
        logger.info(f"[IntentArbiter] Decomposed instruction into {len(res.stages)} stages.")
        return res

    def get_current_stage(self) -> Optional[DecomposedStage]:
        if not self.current_decomposition or not self.current_decomposition.stages:
            return None
        if 0 <= self.current_stage_idx < len(self.current_decomposition.stages):
            return self.current_decomposition.stages[self.current_stage_idx]
        return None

    def advance_stage(self) -> bool:
        """Advances to next sequential stage. Returns False if end of plan reached."""
        if not self.current_decomposition:
            return False
        if self.current_stage_idx + 1 < len(self.current_decomposition.stages):
            self.current_stage_idx += 1
            return True
        elif self.current_decomposition.is_cyclical:
            self.current_stage_idx = 0
            return True
        return False

    def branch_to_stage(self, stage_id: int) -> bool:
        """Branches to specified stage_id upon condition trigger."""
        if not self.current_decomposition:
            return False
        for idx, stage in enumerate(self.current_decomposition.stages):
            if stage.stage_id == stage_id:
                self.current_stage_idx = idx
                logger.info(f"[IntentArbiter] Branched to stage_id {stage_id} at index {idx}")
                return True
        return False

    async def arbitrate(
        self,
        instruction_raw: str,
        rgb_image: Optional[np.ndarray],
        scene_graph_text: str,
        telemetry: DroneTelemetry,
        history: List[Subgoal],
    ) -> Subgoal:
        """Arbitrates the next Subgoal based on current multimodal context."""
        current_stage = self.get_current_stage()
        stage_desc = (
            f"Stage #{current_stage.stage_id}: {current_stage.text}"
            if current_stage
            else f"Full Mission: {instruction_raw}"
        )
        active_queries = current_stage.detector_queries if current_stage else []

        history_str = "\n".join(
            f"  - Subgoal #{s.subgoal_id}: Intent={s.intent}, Waypoint={s.waypoint_xyz}"
            for s in history[-4:]
        ) or "  (None - initial step)"

        user_prompt = (
            f"MISSION CONTEXT:\n"
            f"Raw Instruction: \"{instruction_raw}\"\n"
            f"Active Sub-Stage: {stage_desc}\n\n"
            f"UAV TELEMETRY:\n"
            f"Position: X={telemetry.position[0]:.2f}m, Y={telemetry.position[1]:.2f}m, Z={telemetry.altitude:.2f}m\n"
            f"Heading: {math.degrees(telemetry.yaw):.1f}° | Speed: {telemetry.speed:.2f}m/s\n\n"
            f"RECENT SUBGOALS EXECUTED:\n{history_str}\n\n"
            f"{scene_graph_text}\n\n"
            f"Arbitrate the next Subgoal now."
        )

        images = [rgb_image] if rgb_image is not None else []
        subgoal = await self.vlm.query_with_voting(
            system_prompt=self.ARBITRATION_SYSTEM,
            user_prompt=user_prompt,
            images=images,
            schema=Subgoal,
        )

        # Enforce unique subgoal ID, monotonic timestamp, and active detector queries
        subgoal = subgoal.model_copy(update={
            "subgoal_id": self.subgoal_counter,
            "issued_at": telemetry.timestamp,
            "detector_queries": active_queries or subgoal.detector_queries,
        })
        self.subgoal_counter += 1
        return subgoal
