"""
Target Memory Tracker V6
========================
High-rate visual target tracker, boundary edge detection, and closed-loop
3D metric re-acquisition (completely replacing legacy blind coasting).
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Tuple
import numpy as np

from contracts import (
    TargetRecoveryState,
    TargetRecoveryAction,
    DroneTelemetry,
    SceneNode,
)
from core.scene_graph_3d import MetricSceneGraph3D

logger = logging.getLogger("MavicVLM")


class TargetMemoryTrackerV6:
    """Maintains active target state, anticipates edge departures, and guides 3D recovery."""

    def __init__(
        self,
        edge_threshold: float = 0.70,     # Normalized threshold [-1.0, 1.0] to trigger lead centering
        search_step_budget: int = 6,      # Maximum angular search sectors before declaring lost
        reacquisition_speed: float = 2.5, # Speed to fly toward estimated coordinates (m/s)
    ):
        self.edge_threshold = edge_threshold
        self.search_step_budget = search_step_budget
        self.reacquisition_speed = reacquisition_speed

        self.state = TargetRecoveryState.TRACKING
        self.active_target_id: Optional[int] = None
        self.last_known_centroid_3d: Optional[np.ndarray] = None
        self.search_steps_taken = 0
        self.last_seen_timestamp = 0.0

    def set_active_target(self, node_id: int, scene_graph: MetricSceneGraph3D) -> None:
        """Sets the active target node to track from the 3D scene graph."""
        self.active_target_id = node_id
        if node_id in scene_graph.nodes:
            self.last_known_centroid_3d = scene_graph.nodes[node_id].centroid_world.copy()
            self.state = TargetRecoveryState.TRACKING
            self.search_steps_taken = 0

    def update(
        self,
        target_visible: bool,
        bbox_norm: Optional[Tuple[float, float, float, float]], # [u_min, v_min, u_max, v_max] in [0, 1]
        scene_graph: MetricSceneGraph3D,
        telemetry: DroneTelemetry,
    ) -> TargetRecoveryAction:
        """Evaluates visual tracking status and emits active reaction.
        
        Zero Translation Lock: when target is lost, UAV actively flies toward last known 3D coordinates.
        """
        # If target has a Scene Graph node, update last known coordinates
        if self.active_target_id and self.active_target_id in scene_graph.nodes:
            self.last_known_centroid_3d = scene_graph.nodes[self.active_target_id].centroid_world.copy()

        if target_visible and bbox_norm is not None:
            self.last_seen_timestamp = telemetry.timestamp
            self.search_steps_taken = 0

            # Convert bbox to normalized center [-1.0, 1.0]
            cx_norm = (bbox_norm[0] + bbox_norm[2]) - 1.0
            cy_norm = (bbox_norm[1] + bbox_norm[3]) - 1.0

            # 1. Edge proximity anticipation
            if abs(cx_norm) > self.edge_threshold or abs(cy_norm) > self.edge_threshold:
                self.state = TargetRecoveryState.EDGE_ANTICIPATION
                yaw_rate_lead = -1.2 * cx_norm
                pitch_lead = max(0.0, min(0.8, telemetry.gimbal_pitch + 0.5 * cy_norm))
                return TargetRecoveryAction(
                    gimbal_pitch_cmd=pitch_lead,
                    gimbal_yaw_cmd=0.0,
                    yaw_rate_cmd=yaw_rate_lead,
                    velocity_cmd=None, # Allow normal velocity forward
                    trigger_replan=False,
                )

            # 2. Normal tracking: smooth centering
            self.state = TargetRecoveryState.TRACKING
            return TargetRecoveryAction(
                gimbal_pitch_cmd=telemetry.gimbal_pitch,
                gimbal_yaw_cmd=0.0,
                yaw_rate_cmd=-0.8 * cx_norm,
                velocity_cmd=None,
                trigger_replan=False,
            )

        # Target NOT visible: if no active target assigned, return neutral
        if self.active_target_id is None:
            return TargetRecoveryAction(
                gimbal_pitch_cmd=telemetry.gimbal_pitch,
                gimbal_yaw_cmd=0.0,
                yaw_rate_cmd=0.0,
                velocity_cmd=None,
                trigger_replan=False,
            )

        # Target NOT visible: Execute safe 3D Re-acquisition toward standoff region
        if self.last_known_centroid_3d is not None:
            # Memory staleness check: if target not seen for > 15s, do not blindly assume coordinates
            age = telemetry.timestamp - self.last_seen_timestamp if self.last_seen_timestamp > 0 else 0.0
            if age > 15.0:
                self.state = TargetRecoveryState.LOST_DECLARED
                logger.warning(f"[TargetMemoryTracker] Memory stale (age: {age:.1f}s > 15s). Hovering and requesting replan.")
                return TargetRecoveryAction(
                    gimbal_pitch_cmd=0.2,
                    gimbal_yaw_cmd=0.0,
                    yaw_rate_cmd=0.0,
                    velocity_cmd=np.zeros(3),
                    trigger_replan=True,
                )

            # Retrieve observed dimensions of the object from scene graph if available
            target_extent = np.array([4.0, 4.0, 4.0], dtype=np.float64)
            if self.active_target_id in scene_graph.nodes:
                target_extent = scene_graph.nodes[self.active_target_id].dimensions_3d

            # Compute safe stand-off distance outside the physical obstacle
            half_span = 0.5 * max(float(target_extent[0]), float(target_extent[1]))
            standoff_margin = 3.5  # meters clearance beyond visible surface extent
            d_standoff = half_span + standoff_margin

            diff = self.last_known_centroid_3d - telemetry.position
            dist_to_centroid = float(np.linalg.norm(diff))

            # Direction vector from UAV to object centroid
            if dist_to_centroid > 1e-3:
                unit_vec = diff / dist_to_centroid
            else:
                unit_vec = np.array([1.0, 0.0, 0.0])

            # Standoff waypoint: situated strictly OUTSIDE the object perimeter
            standoff_waypoint = self.last_known_centroid_3d - unit_vec * d_standoff
            diff_to_standoff = standoff_waypoint - telemetry.position
            dist_to_standoff = float(np.linalg.norm(diff_to_standoff))

            # If UAV is outside the standoff perimeter, fly toward the safe observation standoff
            if dist_to_centroid > d_standoff + 1.0:
                self.state = TargetRecoveryState.METRIC_3D_REACQUISITION
                unit_to_standoff = diff_to_standoff / max(1e-3, dist_to_standoff)
                v_reacq = unit_to_standoff * self.reacquisition_speed

                # Camera points directly at estimated object centroid
                dx, dy, dz = diff[0], diff[1], diff[2]
                desired_cam_yaw = math.atan2(dy, dx)
                yaw_diff = (desired_cam_yaw - telemetry.yaw + math.pi) % (2.0 * math.pi) - math.pi
                yaw_rate_cmd = float(np.clip(yaw_diff * 1.5, -1.0, 1.0))

                dist_horiz = math.hypot(dx, dy)
                desired_pitch = float(np.clip(-math.atan2(dz, max(1.0, dist_horiz)), 0.0, 0.8))

                logger.info(
                    f"[TargetMemoryTracker] Safe re-acquisition toward standoff at {standoff_waypoint[:2]} "
                    f"(target centroid: {self.last_known_centroid_3d[:2]}, standoff dist: {d_standoff:.1f}m)."
                )
                return TargetRecoveryAction(
                    gimbal_pitch_cmd=desired_pitch,
                    gimbal_yaw_cmd=0.0,
                    yaw_rate_cmd=yaw_rate_cmd,
                    velocity_cmd=v_reacq,
                    trigger_replan=False,
                )

            # At or within safe standoff perimeter -> Do NOT fly closer into the obstacle!
            # Phase 2: Budgeted Local Scan from safe standoff observation position
            if self.search_steps_taken < self.search_step_budget:
                self.state = TargetRecoveryState.BUDGETED_SEARCH
                self.search_steps_taken += 1
                logger.info(f"[TargetMemoryTracker] Budgeted search sweep step {self.search_steps_taken}/{self.search_step_budget}")
                return TargetRecoveryAction(
                    gimbal_pitch_cmd=0.35,
                    gimbal_yaw_cmd=0.0,
                    yaw_rate_cmd=0.8, # Rotate in increments to reacquire visually
                    velocity_cmd=np.zeros(3),
                    trigger_replan=False,
                )

        # Phase 3: Search budget exhausted or no valid target coordinates -> Declare Lost & Replan
        self.state = TargetRecoveryState.LOST_DECLARED
        logger.warning("[TargetMemoryTracker] Target lost declared. Budget exhausted. Signaling System 2 replanning.")
        return TargetRecoveryAction(
            gimbal_pitch_cmd=0.2,
            gimbal_yaw_cmd=0.0,
            yaw_rate_cmd=0.0,
            velocity_cmd=np.zeros(3),
            trigger_replan=True,
        )
