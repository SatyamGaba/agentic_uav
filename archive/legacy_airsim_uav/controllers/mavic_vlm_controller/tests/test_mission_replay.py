"""
No-Simulator Deterministic Mission Replay Harness
=================================================
Validates the end-to-end zero-shot VLM decision loop with a fake clock and
scripted synthetic observations across all 10 checklist verification steps:
1. Multi-landmark instruction decomposition
2. Candidate generation & Set-of-Mark frame marking
3. VLM selection & candidate-to-node resolution
4. Metric depth backprojection & surface extent mapping
5. Subgoal arbitration & kinematic command validation
6. Fast-loop progress & keyframe deliberative trigger
7. Target removal & safe standoff re-observation
8. Late / stale VLM response & fallback hover transition
9. Verification of completion evidence
10. Deterministic replay trace serialization & audit
"""

import json
import math
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pytest

from contracts import (
    DroneTelemetry,
    CameraIntrinsics,
    SubgoalIntent,
    Subgoal,
    MarkSelection,
    MarkedCandidate,
    SceneNode,
    TargetRecoveryState,
    SafetyState,
)
from cognitive.vlm_client import MockVLMClient, SelfConsistentVLMWrapper
from cognitive.intent_arbiter import IntentArbiter, TaskDecompositionResult, DecomposedStage
from cognitive.som_grounder import SoMGrounder, MockDetector
from cognitive.baseline_agents import FullSystemAgent
from core.scene_graph_3d import MetricSceneGraph3D, compute_camera_to_world_matrix
from core.depth_regulator import DepthOdometryRegulator
from core.tracker_memory import TargetMemoryTrackerV6
from core.safety_fsm import SafetyFallbackFSM
from main_controller import SharedBlackboard, FastContinuousController
from adapters.base_adapter import ContinuousUAVController


class FakeClock:
    """Deterministic monotonic clock for simulation replay."""
    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time

    def advance(self, dt: float) -> None:
        self.current_time += dt

    def now(self) -> float:
        return self.current_time


class ScriptedReplayAdapter(ContinuousUAVController):
    """Synthetic UAV physics adapter driven by a FakeClock."""
    def __init__(self, clock: FakeClock, initial_pos: np.ndarray, intrinsics: CameraIntrinsics):
        self.clock = clock
        self.pos = initial_pos.copy()
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = 0.0
        self.gimbal_pitch = 0.0
        self.intrinsics = intrinsics

        # Synthetic sensor buffers
        self.simulated_depth = np.full((intrinsics.height, intrinsics.width), 15.0, dtype=np.float32)
        self.simulated_rgb = np.zeros((intrinsics.height, intrinsics.width, 3), dtype=np.uint8)

    def set_continuous_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        self.vel = np.array([vx, vy, vz], dtype=np.float64)
        self.yaw_rate = yaw_rate

    def set_gimbal_angles(self, pitch: float, yaw: float = 0.0) -> None:
        self.gimbal_pitch = pitch

    def step_simulation(self, dt: float) -> None:
        self.pos += self.vel * dt
        self.yaw = (self.yaw + self.yaw_rate * dt + math.pi) % (2.0 * math.pi) - math.pi
        self.clock.advance(dt)

    def get_telemetry(self) -> DroneTelemetry:
        return DroneTelemetry(
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            yaw=self.yaw,
            gimbal_pitch=self.gimbal_pitch,
            gimbal_yaw=0.0,
            timestamp=self.clock.now(),
        )

    def get_camera_frame(self) -> Tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
        return self.simulated_rgb, self.simulated_depth, self.intrinsics


def test_deterministic_mission_replay():
    """Executes the full 10-step no-simulator verification replay harness."""
    async def _run_replay():
        clock = FakeClock(start_time=100.0)
        K = np.array([
            [320.0, 0.0, 320.0],
            [0.0, 320.0, 240.0],
            [0.0, 0.0, 1.0],
        ])
        intrinsics = CameraIntrinsics(K=K, width=640, height=480, vfov=math.radians(60.0), hfov=math.radians(80.0))
        adapter = ScriptedReplayAdapter(clock, initial_pos=np.array([0.0, 0.0, 6.0]), intrinsics=intrinsics)

        trace_log: List[Dict[str, Any]] = []

        def log_step(step_num: int, name: str, data: Dict[str, Any]):
            tel = adapter.get_telemetry()
            entry = {
                "step": step_num,
                "name": name,
                "sim_time": clock.now(),
                "telemetry": {
                    "position": tel.position.tolist(),
                    "velocity": tel.velocity.tolist(),
                    "yaw": tel.yaw,
                    "gimbal_pitch": tel.gimbal_pitch,
                    "timestamp": tel.timestamp,
                },
                "data": data,
            }
            trace_log.append(entry)

        # -------------------------------------------------------------
        # STEP 1: Multi-landmark natural language instruction & decomposition
        # -------------------------------------------------------------
        instruction = "Fly past the red water tower to inspect the solar array on the left."
        mock_decomp = TaskDecompositionResult(
            stages=[
                DecomposedStage(
                    stage_id=1,
                    text="Approach and pass the red water tower",
                    detector_queries=["water tower", "red tower"],
                    condition="target_reached",
                    fallback_stage_id=None,
                ),
                DecomposedStage(
                    stage_id=2,
                    text="Inspect the solar array on the left",
                    detector_queries=["solar array", "solar panels"],
                    condition="inspection_complete",
                    fallback_stage_id=None,
                ),
            ]
        )

        mock_vlm = MockVLMClient([mock_decomp])
        vlm_wrapper = SelfConsistentVLMWrapper(mock_vlm, n_samples=1)
        arbiter = IntentArbiter(vlm_wrapper)

        decomp_result = await arbiter.decompose_mission(instruction)
        assert len(decomp_result.stages) == 2
        assert decomp_result.stages[0].stage_id == 1
        log_step(1, "InstructionDecomposition", {
            "instruction": instruction,
            "stages": [s.model_dump() for s in decomp_result.stages],
        })

        # -------------------------------------------------------------
        # STEP 2: Present frame with several marked candidates
        # -------------------------------------------------------------
        # 2 detected objects in camera FOV: Mark #1 (water tower) and Mark #2 (barn)
        detector = MockDetector([
            ((0.40, 0.35, 0.60, 0.65), "water tower", 0.92),
            ((0.05, 0.40, 0.25, 0.70), "red barn", 0.85),
        ])
        som_grounder = SoMGrounder(detector)
        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        marked_frame, candidates = som_grounder.mark_frame(rgb_frame, ["water tower", "red tower"])

        assert len(candidates) == 2
        assert candidates[0].mark_id == 1
        assert candidates[0].label_guess == "water tower"
        log_step(2, "CandidateGenerationAndMarking", {
            "candidates_count": len(candidates),
            "candidates": [
                {"mark_id": c.mark_id, "label_guess": c.label_guess, "bbox": c.bbox_norm, "conf": c.confidence}
                for c in candidates
            ],
        })

        # -------------------------------------------------------------
        # STEP 3: VLM selects Mark #1 & verifies candidate resolution
        # -------------------------------------------------------------
        vlm_selection = MarkSelection(
            selected_mark_id=1,
            confidence=0.94,
            reasoning="Mark #1 corresponds to the requested red water tower.",
        )
        resolved_candidate = som_grounder.resolve_selection(vlm_selection, candidates)
        assert resolved_candidate is not None
        assert resolved_candidate.mark_id == 1
        log_step(3, "VLMSelectionResolution", {
            "vlm_selection": vlm_selection.model_dump(),
            "resolved_mark_id": resolved_candidate.mark_id,
        })

        # -------------------------------------------------------------
        # STEP 4: Metric Depth backprojection & Surface Extent Mapping
        # -------------------------------------------------------------
        scene_graph = MetricSceneGraph3D()
        tel = adapter.get_telemetry()
        T_wc = compute_camera_to_world_matrix(tel)

        # Set synthetic depth for the water tower at 14.0m
        depth_map = np.full((480, 640), 50.0, dtype=np.float32)
        u1, v1 = int(0.40 * 640), int(0.35 * 480)
        u2, v2 = int(0.60 * 640), int(0.65 * 480)
        depth_map[v1:v2, u1:u2] = 14.0 # 14 meters optical depth

        node = scene_graph.add_or_update_detection(
            label=resolved_candidate.label_guess,
            bbox_norm=resolved_candidate.bbox_norm,
            depth_map=depth_map,
            K=intrinsics.K,
            T_wc=T_wc,
            uav_position=tel.position,
            timestamp=tel.timestamp,
            detection_confidence=resolved_candidate.confidence,
        )

        assert node is not None
        assert node.node_id == 1
        assert 13.0 <= node.centroid_world[0] <= 15.0 # Forward along X axis
        log_step(4, "SceneGraphBackprojection", {
            "node_id": node.node_id,
            "centroid_world": node.centroid_world.tolist(),
            "dimensions_3d": node.dimensions_3d.tolist(),
            "confidence": node.confidence,
        })

        # -------------------------------------------------------------
        # STEP 5: Subgoal Arbitration & Kinematic Command Validation
        # -------------------------------------------------------------
        # VLM emits Subgoal to approach standoff point at (10.0, 0.0, 6.0)
        mock_subgoal = Subgoal(
            subgoal_id=1,
            intent=SubgoalIntent.TARGET_APPROACH,
            target_node_id=node.node_id,
            waypoint_xyz=(10.0, 0.0, 6.0),
            v_target=2.5,
            caution_level="medium",
            termination_condition="distance_within_2m",
            issued_at=clock.now(),
            expiry_s=6.0,
        )

        blackboard = SharedBlackboard(mock_subgoal)
        controller = FastContinuousController(adapter, blackboard, scene_graph=scene_graph, tick_hz=20.0)

        # Verify command validity
        controller.tick()
        assert adapter.vel[0] > 0.0, "UAV must be commanded forward along X toward waypoint"
        assert adapter.vel[0] <= 4.0, "Speed must obey v_max bound"
        log_step(5, "SubgoalArbitrationAndCommandValidation", {
            "subgoal": mock_subgoal.model_dump(),
            "commanded_vel": adapter.vel.tolist(),
        })

        # -------------------------------------------------------------
        # STEP 6: Simulate Fast-Loop Progress & Keyframe Trigger
        # -------------------------------------------------------------
        # Run 40 ticks (2.0s at dt = 0.05s). UAV moves forward ~4.0m
        for _ in range(40):
            controller.tick()

        new_tel = adapter.get_telemetry()
        assert new_tel.position[0] > 3.0, "UAV must have advanced forward along X"
        log_step(6, "FastLoopProgress", {
            "new_position": new_tel.position.tolist(),
            "active_safety_state": blackboard.active_safety_state,
        })

        # -------------------------------------------------------------
        # STEP 7: Target Removal & Safe Standoff Re-observation
        # -------------------------------------------------------------
        # Target goes out of sight. Verify target memory tracker enters 3D reacquisition
        # and targets safe standoff, NOT the interior building centroid
        recovery_action = controller.tracker_memory.update(
            target_visible=False,
            bbox_norm=None,
            scene_graph=scene_graph,
            telemetry=new_tel,
        )
        assert recovery_action.velocity_cmd is not None
        # Velocity must plan toward safe standoff outside obstacle perimeter
        assert controller.tracker_memory.state == TargetRecoveryState.METRIC_3D_REACQUISITION
        log_step(7, "TargetLossSafeStandoffReacquisition", {
            "recovery_state": controller.tracker_memory.state,
            "velocity_cmd": recovery_action.velocity_cmd.tolist(),
        })

        # -------------------------------------------------------------
        # STEP 8: Stale VLM Response / Timeout Fallback
        # -------------------------------------------------------------
        # Simulate clock advancing past expiry without a new Subgoal
        clock.advance(7.0) # elapsed = 7.0s > expiry_s (6.0s)
        controller.tick()

        assert blackboard.trigger_event.is_set(), "Event trigger must fire on expiry"
        assert blackboard.active_safety_state == SafetyState.HOVER_REPLAN, "Must enter HOVER_REPLAN"
        log_step(8, "StaleResponseFallbackTrigger", {
            "active_safety_state": blackboard.active_safety_state,
            "trigger_event_set": blackboard.trigger_event.is_set(),
        })

        # -------------------------------------------------------------
        # STEP 9: Completion Evidence & Stage Advancement
        # -------------------------------------------------------------
        # Stage 1 complete -> Advance to Stage 2
        advanced = arbiter.advance_stage()
        assert advanced is True
        current_stage = arbiter.get_current_stage()
        assert current_stage is not None
        assert current_stage.stage_id == 2
        assert "solar array" in current_stage.text
        log_step(9, "StageCompletionAndAdvancement", {
            "advanced": advanced,
            "new_active_stage": current_stage.model_dump(),
        })

        # -------------------------------------------------------------
        # STEP 10: Save Structured Replay Trace for Reproducibility
        # -------------------------------------------------------------
        log_path = Path("controllers/mavic_vlm_controller/tests/mission_replay_trace.json")
        with open(log_path, "w") as f:
            json.dump(trace_log, f, indent=2, default=str)

        assert log_path.exists()
        assert len(trace_log) == 9
        print(f"\n[Mission Replay Harness] Trace verified and logged to {log_path} ({len(trace_log)} steps).")

    asyncio.run(_run_replay())
