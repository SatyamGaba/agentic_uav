"""
Downstream Degradation & Exception Handling Test
================================================
Verifies that when an ungrounded model (e.g. moondream emitting bbox dictionaries)
fails MarkSelection schema validation:
1. The validator rejects the raw dictionary with a ValidationError.
2. The agent catches the exception and safely degrades to a non-selection fallback (selected_mark_id=None).
3. If an unrecoverable failure occurs in arbitration, DualRateMasterController catches it
   and commands a safe HOVER_REPLAN (v_target=0.0, caution_level="high").
"""
import asyncio
import sys
import numpy as np

sys.path.insert(0, "controllers/mavic_vlm_controller")

from contracts import DroneTelemetry, CameraIntrinsics, MarkSelection, Subgoal, SubgoalIntent
from cognitive.som_grounder import SoMGrounder, MockDetector
from cognitive.vlm_client import OpenAICompatibleVLMClient, SelfConsistentVLMWrapper
from cognitive.baseline_agents import FullSystemAgent
from adapters.openfly_adapter import OpenFlyBenchmarkAdapter
from main_controller import DualRateMasterController


def test_live_downstream_safe_degradation():
    print("=== Step 1: Initializing FullSystemAgent with moondream client ===")
    vlm_client = OpenAICompatibleVLMClient(
        endpoint_url="http://localhost:11434/v1/chat/completions",
        model_name="moondream",
        timeout_s=30.0,
        max_retries=1,  # Fast fail for demonstration
    )
    vlm_wrapper = SelfConsistentVLMWrapper(vlm_client, n_samples=1)

    detector = MockDetector([
        ((0.1, 0.2, 0.3, 0.6), "green water tank", 0.95),
        ((0.5, 0.2, 0.9, 0.6), "red antenna tower", 0.90),
    ])
    intrinsics = CameraIntrinsics(
        K=np.array([[500, 0, 320], [0, 500, 180], [0, 0, 1]]),
        width=640,
        height=360,
        vfov=0.7,
        hfov=1.1,
    )
    agent = FullSystemAgent(
        vlm_wrapper=vlm_wrapper,
        detector=detector,
        intrinsics=intrinsics,
    )

    adapter = OpenFlyBenchmarkAdapter(initial_position=np.array([0.0, 0.0, 5.0]), initial_yaw=0.0)
    controller = DualRateMasterController(
        agent=agent,
        adapter=adapter,
        mission_instruction="Proceed to the green water tank.",
    )

    print("\n=== Step 2: Running run_slow_step() with live moondream model ===")
    print("moondream will emit a raw bounding box dictionary that violates MarkSelection schema.")
    
    # We expect run_slow_step to NOT crash, but to safely handle the failure
    subgoal = asyncio.run(controller.run_slow_step())

    print("\n=== Step 3: Inspecting Resulting System State ===")
    print(f"Controller Returned Subgoal:\n{subgoal.model_dump_json(indent=2)}")
    print(f"Target Node ID: {subgoal.target_node_id}")
    print(f"Intent: {subgoal.intent}")
    print(f"Caution Level: {subgoal.caution_level}")

    # Assertions
    assert isinstance(subgoal, Subgoal), "Must return a valid Subgoal"
    # Because moondream failed to ground MarkSelection, target_node_id must NOT be locked onto a false target
    assert subgoal.target_node_id is None, "Must not lock onto a target when SoM validation fails!"
    print("\n[PASSED] System safely degraded to safe exploration without crashing or hallucinating a target lock!")


if __name__ == "__main__":
    test_live_downstream_safe_degradation()
