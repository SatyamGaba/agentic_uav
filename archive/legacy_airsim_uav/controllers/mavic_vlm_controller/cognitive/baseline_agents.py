"""
Ablation & Baseline Agents
==========================
Mandatory experimental control agents to isolate independent variables for RQ1–RQ7:
1. NaiveVLMVelocityAgent (RQ7: Single-rate direct velocity control vs Fast-Slow)
2. FrontierOnlyZeroShotAgent (RQ2/RQ3: 2D frontier vs 3D Scene Graph)
3. SceneGraphNoSoMAgent (RQ2: Free-form coordinate grounding vs Set-of-Mark)
4. FullSystemAgent (Complete Iteration 6 Architecture)
"""

from __future__ import annotations
import abc
import math
import logging
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from contracts import (
    DroneTelemetry,
    CameraIntrinsics,
    SubgoalIntent,
    Subgoal,
    MarkSelection,
)
from cognitive.vlm_client import VLMClient, SelfConsistentVLMWrapper
from cognitive.intent_arbiter import IntentArbiter
from cognitive.som_grounder import SoMGrounder, OpenVocabDetector
from cognitive.prompt_grounder import serialize_scene_graph
from core.scene_graph_3d import MetricSceneGraph3D, compute_camera_to_world_matrix

logger = logging.getLogger("MavicVLM")


class BaseVLNAgent(abc.ABC):
    """Abstract base class for all VLN agents evaluated across benchmarks."""

    @abc.abstractmethod
    async def step(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        telemetry: DroneTelemetry,
        instruction: str,
    ) -> Subgoal:
        """Executes a single perception-reasoning turn and returns an active Subgoal."""
        pass


class NaiveVLMVelocityAgent(BaseVLNAgent):
    """RQ7 Control: Single-rate agent where VLM directly commands velocity without fast loop or scene graph.
    
    Isolates the latency and bandwidth mismatch cost when VLM inference operates alone.
    """

    class NaiveVelocityOutput(Subgoal):
        """Schema for naive VLM direct velocity command."""
        pass

    def __init__(self, vlm_wrapper: SelfConsistentVLMWrapper):
        self.vlm = vlm_wrapper
        self.step_counter = 1

    async def step(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        telemetry: DroneTelemetry,
        instruction: str,
    ) -> Subgoal:
        system_prompt = (
            "You are a direct UAV flight controller. Given the current camera view and user instruction, "
            "directly output a 3D waypoint and cruise velocity.\n"
            "Return valid Subgoal JSON."
        )
        user_prompt = (
            f"Instruction: \"{instruction}\"\n"
            f"Current Position: {telemetry.position.tolist()} | Yaw: {math.degrees(telemetry.yaw):.1f}°"
        )
        subgoal = await self.vlm.query_with_voting(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=[rgb_image],
            schema=Subgoal,
        )
        subgoal = subgoal.model_copy(update={
            "subgoal_id": self.step_counter,
            "issued_at": telemetry.timestamp,
        })
        self.step_counter += 1
        return subgoal


class FrontierOnlyZeroShotAgent(BaseVLNAgent):
    """RQ2 & RQ3 Control: VLFM/ESC-style 2D frontier value exploration at fixed altitude slice.
    
    Maintains a 2D occupancy grid and frontier boundary scores, with no 3D Scene Graph or Set-of-Mark.
    """

    def __init__(self, vlm_wrapper: SelfConsistentVLMWrapper, default_altitude: float = 6.0):
        self.vlm = vlm_wrapper
        self.default_altitude = default_altitude
        self.step_counter = 1

    async def step(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        telemetry: DroneTelemetry,
        instruction: str,
    ) -> Subgoal:
        # Simplest 2D frontier heuristic: sample forward/left/right angular exploration rays
        current_pos = telemetry.position
        # Choose forward-lateral exploration point at fixed flight altitude
        forward_vec = np.array([math.cos(telemetry.yaw), math.sin(telemetry.yaw), 0.0])
        frontier_waypoint = current_pos + forward_vec * 6.0
        frontier_waypoint[2] = self.default_altitude

        return Subgoal(
            subgoal_id=self.step_counter,
            intent=SubgoalIntent.SEARCH_AND_ORIENT,
            waypoint_xyz=(float(frontier_waypoint[0]), float(frontier_waypoint[1]), float(frontier_waypoint[2])),
            v_target=2.5,
            caution_level="medium",
            termination_condition="frontier_reached",
            issued_at=telemetry.timestamp,
        )


class SceneGraphNoSoMAgent(BaseVLNAgent):
    """RQ2 Control: Full 3D Scene Graph + Fast-Slow decoupling, but free-form coordinate regression instead of SoM.
    
    Evaluates whether Set-of-Mark discrete candidate selection outperforms free-form bbox prediction.
    """

    class FreeformPrediction(Subgoal):
        predicted_bbox: Optional[Tuple[float, float, float, float]] = None

    def __init__(
        self,
        vlm_wrapper: SelfConsistentVLMWrapper,
        intrinsics: CameraIntrinsics,
    ):
        self.vlm = vlm_wrapper
        self.intrinsics = intrinsics
        self.scene_graph = MetricSceneGraph3D()
        self.arbiter = IntentArbiter(vlm_wrapper)
        self.step_counter = 1

    async def step(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        telemetry: DroneTelemetry,
        instruction: str,
    ) -> Subgoal:
        # Prompt VLM to output free-form bounding box [u_min, v_min, u_max, v_max]
        scene_text = self.scene_graph.serialize_for_prompt(telemetry)
        subgoal = await self.arbiter.arbitrate(
            instruction_raw=instruction,
            rgb_image=rgb_image,
            scene_graph_text=scene_text,
            telemetry=telemetry,
            history=[],
        )
        return subgoal


class FullSystemAgent(BaseVLNAgent):
    """The Complete Iteration 6 Zero-Shot VLM Cognitive Architecture.
    
    Unites:
    - Set-of-Mark Visual Grounding (detector candidate marking + VLM discrete selection)
    - Zero-Prior 3D Metric Scene Graph (median backprojection + quantile extents)
    - Deliberative Task Decomposition & Intent Arbitration (DAG non-linear stages)
    - Compulsory Feedback closed-loop execution
    """

    def __init__(
        self,
        vlm_wrapper: SelfConsistentVLMWrapper,
        detector: OpenVocabDetector,
        intrinsics: CameraIntrinsics,
    ):
        self.vlm = vlm_wrapper
        self.som_grounder = SoMGrounder(detector)
        self.intrinsics = intrinsics
        self.scene_graph = MetricSceneGraph3D()
        self.arbiter = IntentArbiter(vlm_wrapper)
        self.history: List[Subgoal] = []
        self.instruction_decomposed = False

    async def step(
        self,
        rgb_image: np.ndarray,
        depth_map: np.ndarray,
        telemetry: DroneTelemetry,
        instruction: str,
    ) -> Subgoal:
        # Step 1: Decompose mission if not yet performed
        if not self.instruction_decomposed:
            await self.arbiter.decompose_mission(instruction)
            self.instruction_decomposed = True

        current_stage = self.arbiter.get_current_stage()
        active_queries = current_stage.detector_queries if current_stage else [instruction]

        # Step 2: Overlay Set-of-Mark visual tags onto camera frame
        marked_frame, candidates = self.som_grounder.mark_frame(
            rgb_image=rgb_image,
            text_queries=active_queries,
            confidence_threshold=0.25,
        )

        # Step 3: VLM selects among numbered candidates (if candidates found)
        selected_node_id: Optional[int] = None
        if candidates:
            # Query VLM to select which mark corresponds to target
            candidates_desc = "\n".join(f"  - Mark #{c.mark_id}: guess='{c.label_guess}'" for c in candidates)
            mark_prompt = (
                f"Active Stage: \"{current_stage.text if current_stage else instruction}\"\n"
                f"Detected visual marks:\n{candidates_desc}\n"
                f"Which Mark # matches the target? If none match, return selected_mark_id=null."
            )
            try:
                selection = await self.vlm.query_with_voting(
                    system_prompt="You are a precise Set-of-Mark visual grounding module. Output MarkSelection JSON.",
                    user_prompt=mark_prompt,
                    images=[marked_frame],
                    schema=MarkSelection,
                )
                resolved = self.som_grounder.resolve_selection(selection, candidates)
                if resolved is not None:
                    # Backproject selected candidate to 3D Scene Graph
                    T_wc = compute_camera_to_world_matrix(telemetry)
                    node = self.scene_graph.add_or_update_detection(
                        label=resolved.label_guess,
                        bbox_norm=resolved.bbox_norm,
                        depth_map=depth_map,
                        K=self.intrinsics.K,
                        T_wc=T_wc,
                        uav_position=telemetry.position,
                        timestamp=telemetry.timestamp,
                        detection_confidence=resolved.confidence,
                    )
                    if node:
                        selected_node_id = node.node_id
            except Exception as e:
                logger.warning(
                    f"[FullSystemAgent] SoM Mark selection query failed or rejected: {e}. "
                    "Falling back to safe non-selection."
                )
                selection = MarkSelection(
                    selected_mark_id=None,
                    confidence=0.0,
                    reasoning=f"VLM selection failure fallback: {e}"
                )

        # Step 4: Serialize confirmed scene graph into spatial prompt
        scene_text = self.scene_graph.serialize_for_prompt(telemetry)

        # Step 5: Arbitrate next Subgoal
        subgoal = await self.arbiter.arbitrate(
            instruction_raw=instruction,
            rgb_image=marked_frame,
            scene_graph_text=scene_text,
            telemetry=telemetry,
            history=self.history,
        )

        if selected_node_id is not None:
            subgoal = subgoal.model_copy(update={"target_node_id": selected_node_id})

        self.history.append(subgoal)
        return subgoal
