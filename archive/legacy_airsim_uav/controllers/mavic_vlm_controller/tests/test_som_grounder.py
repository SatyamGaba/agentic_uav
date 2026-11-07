"""
Test suite for Set-of-Mark (SoM) Grounder & Prompt Grounding
===========================================================
Verifies:
- Frame marking with high-contrast numbered badges
- Discrete mark selection resolution
- Fallback contract on zero detector proposals
- Spatial prompt serialization bounding
"""

import math
import numpy as np
import pytest
from contracts import (
    DroneTelemetry,
    SceneNode,
    MarkSelection,
)
from cognitive.som_grounder import MockDetector, SoMGrounder
from cognitive.prompt_grounder import (
    compute_relative_bearing_elevation,
    serialize_scene_graph,
)


def test_som_marking_and_selection_resolution():
    synthetic_img = np.zeros((300, 400, 3), dtype=np.uint8)
    detections = [
        ((0.1, 0.2, 0.3, 0.4), "red brick house", 0.92),
        ((0.6, 0.5, 0.8, 0.9), "wind turbine", 0.85),
    ]
    mock_detector = MockDetector(detections)
    grounder = SoMGrounder(mock_detector)

    marked_img, candidates = grounder.mark_frame(synthetic_img, text_queries=["house", "wind turbine"])

    assert len(candidates) == 2
    assert candidates[0].mark_id == 1
    assert candidates[0].label_guess == "red brick house"
    assert candidates[1].mark_id == 2
    assert candidates[1].label_guess == "wind turbine"

    # Test selection resolution
    selection = MarkSelection(selected_mark_id=2, confidence=0.88, reasoning="Selected turbine.")
    resolved = grounder.resolve_selection(selection, candidates)
    assert resolved is not None
    assert resolved.mark_id == 2
    assert resolved.label_guess == "wind turbine"

    # Test selection of None ("none of the above")
    none_sel = MarkSelection(selected_mark_id=None, confidence=0.1, reasoning="Not present.")
    assert grounder.resolve_selection(none_sel, candidates) is None


def test_som_fallback_on_zero_detections():
    """Fallback contract: when detector finds 0 candidates, return unmodified image and empty list."""
    synthetic_img = np.zeros((300, 400, 3), dtype=np.uint8)
    mock_detector = MockDetector([])
    grounder = SoMGrounder(mock_detector)

    marked_img, candidates = grounder.mark_frame(synthetic_img, text_queries=["nonexistent object"])
    assert len(candidates) == 0
    assert np.array_equal(marked_img, synthetic_img)


def test_prompt_grounder_serialization():
    tel = DroneTelemetry(
        position=np.array([10.0, 10.0, 5.0]),
        velocity=np.array([2.0, 0.0, 0.0]),
        yaw=0.0,
    )
    node = SceneNode(
        node_id=1,
        label="clock_tower",
        centroid_world=np.array([30.0, 10.0, 15.0]),
        dimensions_3d=np.array([4.0, 4.0, 20.0]),
        confidence=0.95,
        observation_count=3,
        last_observed_timestamp=10.0,
        first_viewpoint=np.array([0.0, 0.0, 5.0]),
        is_confirmed=True,
    )

    dist, bearing, elev = compute_relative_bearing_elevation(tel, node)
    assert dist == pytest.approx(math.hypot(20.0, 10.0), abs=0.1)
    assert abs(bearing) < 1.0  # Directly ahead on X axis
    assert elev > 0.0          # Higher than UAV

    prompt_text = serialize_scene_graph([node], tel)
    assert "clock_tower" in prompt_text
    assert "Node #01" in prompt_text
    assert "Dim: (4.0m x 4.0m x 20.0m)" in prompt_text
