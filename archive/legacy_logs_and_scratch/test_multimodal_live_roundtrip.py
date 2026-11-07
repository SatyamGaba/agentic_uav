"""
Live End-to-End Multimodal Round-Trip Test
==========================================
Executes a genuine image + SoM marked candidates query through OpenAICompatibleVLMClient
against local Ollama (moondream) to verify end-to-end multimodal payload construction,
response extraction, thinking token handling, and Pydantic validation into MarkSelection.
"""
import asyncio
import sys
import time
import cv2
import numpy as np

sys.path.insert(0, "controllers/mavic_vlm_controller")

from contracts import MarkSelection, MarkedCandidate
from cognitive.som_grounder import SoMGrounder, MockDetector
from cognitive.vlm_client import OpenAICompatibleVLMClient

def test_live_multimodal_som():
    # 1. Synthesize 640x360 aerial frame with two salient landmarks
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    img[:] = (210, 210, 210)  # Neutral grey background
    
    # Landmark 1: Green rooftop water tank (left)
    cv2.rectangle(img, (80, 100), (220, 250), (35, 175, 45), -1)
    # Landmark 2: Red antenna tower (right)
    cv2.rectangle(img, (420, 70), (530, 270), (30, 40, 210), -1)

    # 2. Open-vocabulary detector proposals
    detector = MockDetector([
        ((80/640, 100/360, 220/640, 250/360), "green water tank", 0.94),
        ((420/640, 70/360, 530/640, 270/360), "red antenna tower", 0.91),
    ])
    grounder = SoMGrounder(detector)
    annotated_frame, candidates = grounder.mark_frame(
        img, ["green water tank", "red antenna tower"]
    )
    
    print(f"[Step 1] Synthesized image with {len(candidates)} SoM candidates:")
    for c in candidates:
        print(f"  Candidate [{c.mark_id}]: {c.label_guess} at bbox {c.bbox_norm}")

    # 3. Instantiate production client pointing to Ollama endpoint
    client = OpenAICompatibleVLMClient(
        endpoint_url="http://localhost:11434/v1/chat/completions",
        model_name="moondream",
        timeout_s=30.0,
    )

    system_prompt = (
        "You are an aerial UAV visual navigation assistant. "
        "Analyze the provided image with numbered Set-of-Mark candidates and select the mark that matches the navigation query."
    )
    user_prompt = (
        "Mission query: 'Approach the green water tank on the rooftop'.\n"
        "Which marked candidate ID corresponds to the green water tank?\n"
        "Candidates:\n"
        "- Mark [1]: green water tank\n"
        "- Mark [2]: red antenna tower\n"
        "Return a JSON object with: 'selected_mark_id' (integer 1 or 2), 'confidence' (float), and 'reasoning' (string)."
    )

    # 4. Execute query through actual production client class
    async def _run():
        t0 = time.time()
        result = await client.query_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=[annotated_frame],
            schema=MarkSelection,
            temperature=0.1
        )
        elapsed = time.time() - t0
        return result, elapsed

    result, elapsed = asyncio.run(_run())
    
    print(f"\n[Step 2] Live VLM Response Received in {elapsed:.2f}s:")
    print(f"  Type: {type(result)}")
    print(f"  Parsed Pydantic Object: {result}")
    print(f"  Selected Mark ID: {result.selected_mark_id}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Reasoning: {result.reasoning}")

    # 5. Resolve candidate
    chosen = grounder.resolve_selection(result, candidates)
    print(f"\n[Step 3] Grounder Resolution: {chosen}")
    
    assert isinstance(result, MarkSelection), "Result is not a MarkSelection Pydantic instance"
    print("\n[PASSED] LIVE MULTIMODAL ROUND-TRIP TEST PASSED!")

if __name__ == "__main__":
    test_live_multimodal_som()
