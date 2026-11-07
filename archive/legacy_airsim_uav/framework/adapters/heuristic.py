"""
Heuristic Rule-Based VLN Adapter.
Provides deterministic, zero-dependency baseline navigation for verification, offline testing, and CI.
"""
import re
import math
from typing import Dict, Any, Optional
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.types import VLNObservation, NavAction
from framework.perception.transforms import (
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame
)


class HeuristicRuleAdapter(BaseVLNAdapter):
    """
    Deterministic rule-based baseline adapter.
    Parses spatial cues directly from instruction text and samples visual depth corridors.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="HeuristicRuleAdapter", config=config)
        self.intrinsics = CameraIntrinsics()
        self.step_counter = 0

    def reset(self, initial_observation: VLNObservation) -> None:
        self.step_counter = 0

    def step(self, observation: VLNObservation) -> NavAction:
        self.step_counter += 1
        instruction = observation.instruction.lower()

        # Check for arrival conditions
        if self.step_counter > 20 or "stop" in instruction and self.step_counter > 5:
            # Check if landmark appears close
            depth_center = query_metric_depth_patch(observation.depth_map, 0.5, 0.5)
            if depth_center < 3.5:
                return NavAction(is_stop=True, reasoning="Landmark reached (proximity confirmed)", depth_m=depth_center)

        # Parse steering cues from text
        target_u = 0.5
        yaw_bias = 0.0
        if "turn left" in instruction or "veer left" in instruction or "left" in instruction:
            target_u = 0.3
            yaw_bias = -math.radians(20.0)
        elif "turn right" in instruction or "veer right" in instruction or "right" in instruction:
            target_u = 0.7
            yaw_bias = math.radians(20.0)

        # Climb / descent cues
        dz_bias = 0.0
        if any(w in instruction for w in ["climb", "ascend", "higher", "up"]):
            dz_bias = 2.0
        elif any(w in instruction for w in ["descend", "lower", "down"]):
            dz_bias = -2.0

        # Query metric depth at target corridor
        depth_m = query_metric_depth_patch(observation.depth_map, target_u, 0.5, default_depth=8.0)

        # Back-project to 3D body displacement
        dx, dy, dz, yaw_offset = backproject_pixel_to_body_frame(
            u=target_u,
            v=0.5,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            max_range_m=7.0
        )

        return NavAction(
            dx=dx,
            dy=dy,
            dz=dz + dz_bias,
            yaw_delta=yaw_offset + yaw_bias,
            u=target_u,
            v=0.5,
            depth_m=depth_m,
            is_stop=False,
            reasoning=f"Heuristic corridor tracking (u={target_u:.2f}, depth={depth_m:.1f}m)"
        )
