"""
Live Test of OpenAICompatibleVLMClient with Thinking-Token Stripping
===================================================================
Calls the actual OpenAICompatibleVLMClient against qwen3.5:latest on Ollama.
Qwen 3.5 is a reasoning model that emits <think> ... </think> blocks.
This test proves that the production client class strips the live thinking tokens
and successfully validates and returns an actual Subgoal Pydantic object.
"""
import asyncio
import sys
import time

sys.path.insert(0, "controllers/mavic_vlm_controller")

from contracts import Subgoal, SubgoalIntent
from cognitive.vlm_client import OpenAICompatibleVLMClient

def test_qwen_live_subgoal_parsing():
    client = OpenAICompatibleVLMClient(
        endpoint_url="http://localhost:11434/v1/chat/completions",
        model_name="qwen3.5:latest",
        timeout_s=90.0,
    )

    system_prompt = (
        "You are System 2 of an autonomous UAV cognitive architecture. "
        "Decompose the high-level mission into a structured Subgoal directive. "
        "You must output valid JSON conforming to the Subgoal schema."
    )
    user_prompt = (
        "Mission instruction: 'Survey the southern perimeter of the industrial park at 15m altitude'.\n"
        "Generate the next Subgoal directive for System 1.\n"
        "Fields required:\n"
        "- subgoal_id: integer\n"
        "- intent: one of INTENT_SEARCH_AND_ORIENT, INTENT_CORRIDOR_TRANSIT, INTENT_TARGET_APPROACH, INTENT_SURVEY_INSPECT, INTENT_TERMINAL_VERIFY\n"
        "- waypoint_xyz: [x, y, z] coordinate tuple in meters\n"
        "- v_target: target speed float (e.g. 3.0)\n"
        "- caution_level: 'low', 'medium', or 'high'\n"
        "- termination_condition: string\n"
        "- expiry_s: float (e.g. 6.0)\n"
        "- detector_queries: list of strings (e.g. ['fence', 'perimeter'])\n"
    )

    async def _run():
        t0 = time.time()
        subgoal = await client.query_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=[],
            schema=Subgoal,
            temperature=0.2,
        )
        elapsed = time.time() - t0
        return subgoal, elapsed

    print("Sending live request to qwen3.5:latest via OpenAICompatibleVLMClient...")
    subgoal, elapsed = asyncio.run(_run())
    
    print(f"\n[LIVE CALL SUCCESS in {elapsed:.2f}s]")
    print(f"Returned object type: {type(subgoal)}")
    print(f"Validated Pydantic Subgoal:\n{subgoal.model_dump_json(indent=2)}")
    print(f"Subgoal Intent: {subgoal.intent} (Enum: {isinstance(subgoal.intent, SubgoalIntent)})")
    print(f"Waypoint: {subgoal.waypoint_xyz}")
    print(f"Caution Level: {subgoal.caution_level}")
    print(f"Detector Queries: {subgoal.detector_queries}")

    assert isinstance(subgoal, Subgoal), "Result must be a Subgoal instance"
    assert isinstance(subgoal.intent, SubgoalIntent), "Intent must be SubgoalIntent enum"
    print("\n[PASSED] Real model output with thinking tokens parsed end-to-end into Subgoal!")

if __name__ == "__main__":
    test_qwen_live_subgoal_parsing()
