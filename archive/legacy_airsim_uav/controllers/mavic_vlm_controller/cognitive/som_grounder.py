"""
Set-of-Mark (SoM) Visual Grounder
=================================
Transforms the VLM's perception role from fragile free-form coordinate
regression to robust discrete semantic selection over numbered visual candidates.
"""

from __future__ import annotations
import abc
import logging
from typing import List, Tuple, Optional
import cv2
import numpy as np

from contracts import MarkedCandidate, MarkSelection

logger = logging.getLogger("MavicVLM")


class OpenVocabDetector(abc.ABC):
    """Abstract interface for open-vocabulary 2D object detection."""

    @abc.abstractmethod
    def detect(
        self,
        rgb_image: np.ndarray,
        text_queries: List[str],
        confidence_threshold: float = 0.25,
    ) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        """Detects objects matching text queries.
        
        Returns:
            List of ([u_min, v_min, u_max, v_max], label, confidence)
            where bounding box coordinates are normalized in [0.0, 1.0].
        """
        pass


class MockDetector(OpenVocabDetector):
    """Deterministic Mock detector for testing SoM grounding pipelines."""

    def __init__(self, predefined_detections: Optional[List[Tuple[Tuple[float, float, float, float], str, float]]] = None):
        self.predefined_detections = list(predefined_detections) if predefined_detections else []

    def queue_detections(self, detections: List[Tuple[Tuple[float, float, float, float], str, float]]) -> None:
        self.predefined_detections.extend(detections)

    def detect(
        self,
        rgb_image: np.ndarray,
        text_queries: List[str],
        confidence_threshold: float = 0.25,
    ) -> List[Tuple[Tuple[float, float, float, float], str, float]]:
        # Filter by confidence
        return [d for d in self.predefined_detections if d[2] >= confidence_threshold]


class SoMGrounder:
    """Overlays discrete numbered marks onto visual frames and resolves VLM selections."""

    # High-contrast color palette for visual marks (BGR format)
    PALETTE = [
        (0, 255, 0),    # Bright Green
        (0, 255, 255),  # Yellow
        (255, 100, 0),  # Blue-Cyan
        (0, 165, 255),  # Orange
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
    ]

    def __init__(self, detector: OpenVocabDetector):
        self.detector = detector

    def mark_frame(
        self,
        rgb_image: np.ndarray,
        text_queries: List[str],
        confidence_threshold: float = 0.25,
    ) -> Tuple[np.ndarray, List[MarkedCandidate]]:
        """Detects candidates and overlays high-contrast numbered tags onto the frame."""
        h, w = rgb_image.shape[:2]
        detections = self.detector.detect(rgb_image, text_queries, confidence_threshold)
        
        if not detections:
            # Fallback contract: return unmodified frame and empty candidate list
            return rgb_image.copy(), []

        annotated = rgb_image.copy()
        candidates: List[MarkedCandidate] = []

        for idx, (bbox_norm, label, conf) in enumerate(detections, start=1):
            u1 = int(bbox_norm[0] * w)
            v1 = int(bbox_norm[1] * h)
            u2 = int(bbox_norm[2] * w)
            v2 = int(bbox_norm[3] * h)

            color = self.PALETTE[(idx - 1) % len(self.PALETTE)]

            # 1. Draw bounding box rectangle
            cv2.rectangle(annotated, (u1, v1), (u2, v2), color, 2)

            # 2. Draw high-visibility marker badge [ #ID: label ]
            tag_text = f"[{idx}] {label}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(tag_text, font, font_scale, thickness)

            badge_y1 = max(0, v1 - th - 6)
            badge_y2 = v1
            badge_x2 = min(w, u1 + tw + 8)

            cv2.rectangle(annotated, (u1, badge_y1), (badge_x2, badge_y2), color, -1)
            cv2.putText(annotated, tag_text, (u1 + 4, badge_y2 - 3), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            candidate = MarkedCandidate(
                mark_id=idx,
                bbox_norm=bbox_norm,
                label_guess=label,
                confidence=conf,
            )
            candidates.append(candidate)

        return annotated, candidates

    def resolve_selection(
        self,
        selection: MarkSelection,
        candidates: List[MarkedCandidate],
    ) -> Optional[MarkedCandidate]:
        """Resolves the VLM's discrete choice to the corresponding candidate."""
        if selection.selected_mark_id is None:
            return None
        for c in candidates:
            if c.mark_id == selection.selected_mark_id:
                return c
        logger.warning(f"[SoMGrounder] VLM selected mark_id={selection.selected_mark_id}, but it does not exist.")
        return None
