"""
Comprehensive Test Suite for VLM Client & Self-Consistency Wrapper
===================================================================
Tests all execution paths in vlm_client.py:
- Robust markdown-json and thinking token extraction
- Image base64 encoding and OpenAI payload formatting
- SelfConsistentVLMWrapper voting: SoM discrete consensus, ties, Subgoal intent voting, caution propagation
- Retry loops, timeout handling, and failure resilience
"""
import asyncio
import json
import numpy as np
import pytest
from pydantic import ValidationError

from contracts import MarkSelection, Subgoal, SubgoalIntent
from cognitive.vlm_client import (
    extract_json_from_llm_output,
    MockVLMClient,
    OpenAICompatibleVLMClient,
    SelfConsistentVLMWrapper,
)


def test_extract_json_edge_cases():
    # 1. Strip thinking tokens followed by markdown code fence
    text1 = "<think>Let me reason about this...</think>\n```json\n{\"selected_mark_id\": 2, \"confidence\": 0.85}\n```"
    d1 = extract_json_from_llm_output(text1)
    assert d1["selected_mark_id"] == 2
    assert d1["confidence"] == 0.85

    # 2. Outermost braces without code fence
    text2 = "Here is the chosen landmark: {\"selected_mark_id\": 1, \"confidence\": 0.9, \"reasoning\": \"visible\"} Hope this helps!"
    d2 = extract_json_from_llm_output(text2)
    assert d2["selected_mark_id"] == 1

    # 3. Direct JSON string
    text3 = "{\"selected_mark_id\": 3, \"confidence\": 0.7}"
    d3 = extract_json_from_llm_output(text3)
    assert d3["selected_mark_id"] == 3


def test_openai_client_image_b64_encoding():
    client = OpenAICompatibleVLMClient(model_name="test-vlm")
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    b64 = client._encode_image_b64(img)
    assert isinstance(b64, str)
    assert len(b64) > 50

    payload = client._build_payload(
        system_prompt="sys",
        user_prompt="user",
        images=[img],
        temperature=0.2
    )
    assert payload["model"] == "test-vlm"
    assert payload["response_format"] == {"type": "json_object"}
    assert len(payload["messages"][1]["content"]) == 2  # text + image_url


def test_self_consistency_som_clear_winner():
    # 3 samples: mark 1 (conf 0.8), mark 1 (conf 0.9), mark 2 (conf 0.7)
    mock = MockVLMClient([
        MarkSelection(selected_mark_id=1, confidence=0.8, reasoning="sample1"),
        MarkSelection(selected_mark_id=1, confidence=0.9, reasoning="sample2"),
        MarkSelection(selected_mark_id=2, confidence=0.7, reasoning="sample3"),
    ])
    wrapper = SelfConsistentVLMWrapper(mock, n_samples=3)

    result = asyncio.run(wrapper.query_with_voting("sys", "user", [], MarkSelection))
    assert result.selected_mark_id == 1
    assert pytest.approx(result.confidence, 1e-4) == 0.85  # Average of 0.8 and 0.9


def test_self_consistency_som_tie_rejection():
    # 2 samples in tie: mark 1 vs mark 2 -> must reject and return safe null
    mock = MockVLMClient([
        MarkSelection(selected_mark_id=1, confidence=0.9, reasoning="sample1"),
        MarkSelection(selected_mark_id=2, confidence=0.9, reasoning="sample2"),
    ])
    wrapper = SelfConsistentVLMWrapper(mock, n_samples=2)

    result = asyncio.run(wrapper.query_with_voting("sys", "user", [], MarkSelection))
    assert result.selected_mark_id is None
    assert result.confidence == 0.0


def test_self_consistency_subgoal_intent_voting_and_caution():
    # 3 samples: TARGET_APPROACH (low caution), TARGET_APPROACH (high caution), SURVEY_INSPECT
    sg1 = Subgoal(subgoal_id=1, intent=SubgoalIntent.TARGET_APPROACH, waypoint_xyz=(1, 2, 3), caution_level="low")
    sg2 = Subgoal(subgoal_id=2, intent=SubgoalIntent.TARGET_APPROACH, waypoint_xyz=(1, 2, 3), caution_level="high")
    sg3 = Subgoal(subgoal_id=3, intent=SubgoalIntent.SURVEY_INSPECT, waypoint_xyz=(1, 2, 3), caution_level="low")

    mock = MockVLMClient([sg1, sg2, sg3])
    wrapper = SelfConsistentVLMWrapper(mock, n_samples=3)

    result = asyncio.run(wrapper.query_with_voting("sys", "user", [], Subgoal))
    assert result.intent == SubgoalIntent.TARGET_APPROACH
    # High caution must propagate if any matching sample requested high caution
    assert result.caution_level == "high"


def test_self_consistency_subgoal_intent_tie():
    # 2 samples in tie: TARGET_APPROACH vs SEARCH_AND_ORIENT -> must enforce high caution
    sg1 = Subgoal(subgoal_id=1, intent=SubgoalIntent.TARGET_APPROACH, waypoint_xyz=(1, 2, 3), v_target=3.0, caution_level="low")
    sg2 = Subgoal(subgoal_id=2, intent=SubgoalIntent.SEARCH_AND_ORIENT, waypoint_xyz=(1, 2, 3), v_target=2.0, caution_level="medium")

    mock = MockVLMClient([sg1, sg2])
    wrapper = SelfConsistentVLMWrapper(mock, n_samples=2)

    result = asyncio.run(wrapper.query_with_voting("sys", "user", [], Subgoal))
    assert result.caution_level == "high"
    assert result.v_target <= 1.0


def test_self_consistency_sample_failure_resilience():
    class FailingMock(MockVLMClient):
        def __init__(self):
            super().__init__()
            self.call_idx = 0

        async def query_structured(self, system_prompt, user_prompt, images, schema, temperature=0.2):
            self.call_idx += 1
            if self.call_idx == 1:
                raise ValueError("Simulated network timeout on sample 1")
            return MarkSelection(selected_mark_id=2, confidence=0.9, reasoning="valid")

    mock = FailingMock()
    wrapper = SelfConsistentVLMWrapper(mock, n_samples=2)

    result = asyncio.run(wrapper.query_with_voting("sys", "user", [], MarkSelection))
    assert result.selected_mark_id == 2
    assert result.confidence == 0.9


def test_self_consistency_all_samples_fail():
    class AllFailMock(MockVLMClient):
        async def query_structured(self, system_prompt, user_prompt, images, schema, temperature=0.2):
            raise ConnectionError("Backend unavailable")

    mock = AllFailMock()
    wrapper = SelfConsistentVLMWrapper(mock, n_samples=2)

    with pytest.raises(RuntimeError, match="All 2 self-consistency samples failed"):
        asyncio.run(wrapper.query_with_voting("sys", "user", [], MarkSelection))


def test_mark_selection_strict_validation():
    # Unrecognized dictionary (e.g. raw bbox output from ungrounded model) must raise ValidationError
    bad_dict = {"green water tank": [0.0, 0.3, 0.34, 0.63]}
    with pytest.raises(ValidationError):
        MarkSelection.model_validate(bad_dict)

    # Valid dictionary with selected_mark_id must pass
    good_dict = {"selected_mark_id": 1, "confidence": 0.95, "reasoning": "tank"}
    sel = MarkSelection.model_validate(good_dict)
    assert sel.selected_mark_id == 1
