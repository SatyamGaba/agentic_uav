"""
Depth-Odometry In-Loop Distance Regulator
=========================================
Couples float32 metric depth maps with real-time UAV odometry (speed and acceleration)
to regulate forward clearance and prevent terminal overshoot and collision.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Dict, Any


import logging

logger = logging.getLogger("DepthOdometryRegulator")


class DepthOdometryRegulator:
    """In-loop metric closeness regulator for continuous flight."""

    def __init__(
        self,
        grid_rows: int = 3,
        grid_cols: int = 3,
        d_margin: float = 1.5,
        d_slowdown: float = 8.0,
        t_react: float = 0.1,
        a_brake: float = 2.5,
        alpha: float = 0.7,
    ):
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.d_margin = d_margin
        self.d_slowdown = d_slowdown
        self.t_react = t_react
        self.a_brake = a_brake
        self.alpha = alpha

    def compute_zone_depths(self, depth_map: np.ndarray, max_clearance: float = 100.0) -> np.ndarray:
        """Divides depth map into rows x cols grid and returns 10th percentile depth per zone.
        
        CRITICAL SAFETY RULE:
        1. Open Sky: Rays that miss geometry return sentinels (>100m, 26544m, or +inf). A zone dominated
           by sky represents open airspace and reports max_clearance (100.0m), NOT 0.0m.
        2. Sparse Obstacles: If 1-10 pixels register an obstacle (e.g. thin powerline/pole), the zone
           reports the minimum valid depth to prevent collision.
        3. Near-Contact: If pixels register <= 0.1m, lens occlusion / physical contact is declared (0.0m).
        4. Sensor Dropout: Only when an entire frame or non-sky patch has zero valid/sky data is 0.0m returned.
        """
        h, w = depth_map.shape[:2]
        r_step = h // self.grid_rows
        c_step = w // self.grid_cols

        zones = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        # Global sensor validity check: must have positive or sentinel depth
        has_positive = np.any(depth_map > 0.1)
        if not has_positive:
            # Complete sensor blackout / uninitialized camera
            logger.warning("Global sensor blackout detected: all depth pixels <= 0.1m. Declaring 0.0m clearance.")
            return zones

        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                patch = depth_map[r * r_step:(r + 1) * r_step, c * c_step:(c + 1) * c_step]
                finite_patch = np.isfinite(patch)

                # 1. Near-contact obstacle hazard (lens pressed against structure)
                near_contact = patch[finite_patch & (patch > 0.0) & (patch <= 0.1)]
                if len(near_contact) > 0.1 * patch.size:
                    logger.warning(
                        "Zone (%d, %d): Near-contact hazard detected (%.1f%% pixels <= 0.1m). Declaring 0.0m.",
                        r, c, (len(near_contact) / patch.size) * 100.0
                    )
                    zones[r, c] = 0.0
                    continue

                # 2. Valid obstacle returns within sensing range (0.1m - max_clearance)
                valid = patch[finite_patch & (patch > 0.1) & (patch < max_clearance)]

                if len(valid) > 10:
                    # Sufficient depth points: use 10th percentile
                    zones[r, c] = float(np.percentile(valid, 10))
                elif len(valid) > 0:
                    # Sparse returns: DISTINCTION: This zone has 1-10 valid returns because of a thin physical obstacle
                    # (e.g., antenna, wire, distant foliage) rather than dense surface geometry.
                    # We report the minimum detected depth to enforce conservative safety and avoid collisions.
                    min_d = float(np.min(valid))
                    logger.info(
                        "Zone (%d, %d): Sparse obstacle detected (%d valid pixels, min_depth=%.2fm). Using min depth.",
                        r, c, len(valid), min_d
                    )
                    zones[r, c] = min_d
                else:
                    # Zero valid returns within range:
                    # DISTINCTION: Distinguish genuine open sky (rays cast to infinity returning sentinels >= max_clearance)
                    # from sensor dropout/partial occlusion (invalid NaNs, zeros, or corrupted channels).
                    sky_sentinels = patch[(~finite_patch) | (patch >= max_clearance)]
                    if len(sky_sentinels) > 0.5 * patch.size:
                        # Genuine open sky: rays hit no obstacle within sensor range
                        logger.debug(
                            "Zone (%d, %d): Open-sky fallback triggered (%d/%d sky sentinels). Reporting max clearance %.1fm.",
                            r, c, len(sky_sentinels), patch.size, max_clearance
                        )
                        zones[r, c] = max_clearance
                    else:
                        # Uninterpretable dropout / complete signal loss / black hole occlusion
                        logger.warning(
                            "Zone (%d, %d): Inconclusive sensor dropout (<10 valid, only %d/%d sentinels). Falling back to 0.0m.",
                            r, c, len(sky_sentinels), patch.size
                        )
                        zones[r, c] = 0.0

        return zones

    def frontal_clearance(self, zone_depths: np.ndarray) -> float:
        """Returns minimum clearance across central frontal sectors."""
        # Central row (middle elevation), across left-center, center, right-center
        mid_r = self.grid_rows // 2
        frontal_sector = zone_depths[mid_r, :]
        return float(np.min(frontal_sector))

    def stopping_distance(self, current_speed: float) -> float:
        """Calculates dynamic stopping distance required to brake safely to 0 m/s."""
        v = max(0.0, current_speed)
        d_kinematic = (v * v) / (2.0 * self.a_brake)
        d_reaction = v * self.t_react
        return d_kinematic + d_reaction + self.d_margin

    def scale_forward_velocity(self, v_desired: float, d_frontal: float, current_speed: float = 0.0) -> float:
        """Throttles forward speed based on physical clearance and dynamic braking limits.
        
        Guarantees:
        1. Zero forward speed when d_frontal <= d_margin.
        2. Dynamic braking cap: v_cmd cannot exceed v_safe = sqrt(2 * a_brake * max(0, d_frontal - d_margin)).
        3. Immediate stop if d_frontal <= stopping_distance(current_speed).
        """
        if d_frontal <= self.d_margin:
            return 0.0

        d_stop = self.stopping_distance(current_speed)
        # If available clearance is less than or equal to current stopping distance, hard brake!
        if d_frontal <= d_stop:
            return 0.0

        # Maximum kinematically safe speed that allows coming to a complete stop before d_margin
        safe_clearance = max(0.0, d_frontal - self.d_margin)
        v_kinematic_limit = math.sqrt(2.0 * self.a_brake * safe_clearance)

        effective_slowdown = max(self.d_slowdown, d_stop + 1.0)
        if d_frontal >= effective_slowdown:
            return min(v_desired, v_kinematic_limit)

        # Progressive deceleration curve
        ratio = (d_frontal - self.d_margin) / (effective_slowdown - self.d_margin)
        clamped_ratio = max(0.0, min(1.0, ratio))
        v_scaled = v_desired * (clamped_ratio ** self.alpha)
        return min(v_scaled, v_kinematic_limit)
