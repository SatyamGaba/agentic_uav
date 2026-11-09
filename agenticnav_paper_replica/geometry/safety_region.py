"""Safety corridor evaluation and kinematics quantization (v7 Specification).

Canonical Right-Handed Agent Frame:
    - Forward:  -Z
    - Right:    +X
    - Up:       +Y

Bearing Angle theta:
    theta = atan2(-p_x, -p_z)
    - Straight ahead (p_x=0,  p_z=-D): theta = 0 rad
    - Left           (p_x<0,  p_z=-D): theta > 0 rad (CCW positive)
    - Right          (p_x>0,  p_z=-D): theta < 0 rad (CW negative)

Corridor Vectors:
    Forward unit vector: b_theta = [-sin(theta), -cos(theta)]^T
    Lateral unit vector: l_theta = [ cos(theta), -sin(theta)]^T

Safety Corridor Definition:
    R = { s * b_theta + h * l_theta | 0 < s < rho_q + r_a, |h| < r_a } [P]
"""

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class SafetyCheckResult:
    status: str  # "executed" or "reselect"
    cause: Optional[str] = None  # "invalid_depth", "out_of_range", "obstacle", "no_motion"
    raw_distance_m: float = 0.0
    raw_bearing_deg: float = 0.0
    effective_margin_m: float = 0.0
    quantized_rho_m: float = 0.0
    quantized_theta_deg: float = 0.0
    turn_steps: int = 0
    forward_steps: int = 0
    collision_point_count: int = 0
    snapped_adjacent_bin: bool = False


class GroundCorridorChecker:
    """Rigorous 2D ground corridor safety and kinematic quantization engine (v7)."""

    def __init__(
        self,
        robot_radius: float = 0.10,          # r_a in meters [Calibrated to simulator agent radius 0.10m]
        base_margin: float = 0.30,           # m in meters [?]
        rho_max: float = 2.5,                # max forward step in meters [?]
        turn_step_deg: float = 15.0,         # discrete simulator turn increment [P]
        forward_step_m: float = 0.25,        # discrete simulator forward increment [P]
        zero_rho_policy: str = "rotate",     # "rotate" (default), "reject"
        allow_adjacent_bin_snap: bool = False, # default off in v7 (ablation only)
        max_snap_deg: float = 9.0,
        max_snap_drift_m: float = 0.35,
        obstacle_point_threshold: int = 1,
    ):
        self.r_a = robot_radius
        self.base_margin = base_margin
        self.rho_max = rho_max
        self.turn_step_deg = turn_step_deg
        self.forward_step_m = forward_step_m
        self.zero_rho_policy = zero_rho_policy
        self.allow_adjacent_bin_snap = allow_adjacent_bin_snap
        self.max_snap_deg = max_snap_deg
        self.max_snap_drift_m = max_snap_drift_m
        self.obstacle_point_threshold = obstacle_point_threshold

    def compute_kinematics(
        self,
        p_x: float,
        p_z: float,
        m_eff: Optional[float] = None,
    ) -> Tuple[float, float, float, float]:
        """Compute continuous and quantized displacement and bearing.

        Formula from Blueprint v7 Section 5.3:
            D_raw = sqrt(p_x^2 + p_z^2)
            theta = atan2(-p_x, -p_z)
            rho = min(rho_max, max(0.0, D_raw - m)) [P]
            theta_q = turn_step * round(theta / turn_step)
            rho_q = forward_step * floor(rho / forward_step)
        """
        if m_eff is None:
            m_eff = self.base_margin

        D_raw = math.sqrt(p_x * p_x + p_z * p_z)
        theta_rad = math.atan2(-p_x, -p_z)
        theta_deg = math.degrees(theta_rad)

        # Displacement formula [P]
        rho_cont = min(self.rho_max, max(0.0, D_raw - m_eff))

        # Strict floor quantization
        rho_q = self.forward_step_m * math.floor(rho_cont / self.forward_step_m)

        # Discrete turn quantization
        theta_q_deg = self.turn_step_deg * round(theta_deg / self.turn_step_deg)
        while theta_q_deg > 180.0:
            theta_q_deg -= 360.0
        while theta_q_deg <= -180.0:
            theta_q_deg += 360.0

        return D_raw, theta_deg, rho_q, theta_q_deg

    def test_corridor_collision(
        self,
        theta_deg: float,
        rho_q: float,
        obstacles_2d: np.ndarray,
        p_y: float = 0.0,
        obstacles_3d: Optional[np.ndarray] = None,
        D_raw: Optional[float] = None,
        step_clearance_m: float = 0.15,
        head_clearance_m: float = 1.25,
    ) -> Tuple[bool, int]:
        """Test whether any obstacle point falls inside the swept safety volume.

        If obstacles_3d (shape (N, 3)) is provided, performs a true 3D swept cylinder
        check along the continuous trajectory vector (s, h, y):
            - s in (0, rho_q + r_a)
            - |h| < r_a
            - y in [y_base(s) + step_clearance_m, y_base(s) + head_clearance_m]
          where y_base(s) = s * (p_y / D_raw) is the continuous elevation of the path
          (handling flat floors, inclines, stairs, and terrain ramps naturally).

        If obstacles_3d is None, falls back to the planar 2D rectangular corridor.
        """
        # 3D Swept Cylinder Mode
        if obstacles_3d is not None and obstacles_3d.shape[0] > 0 and obstacles_3d.shape[1] >= 3:
            ox = obstacles_3d[:, 0]
            oy = obstacles_3d[:, 1]
            oz = obstacles_3d[:, 2]

            # If pure rotation (rho_q == 0), check in-place chassis cylinder
            if rho_q <= 0.0:
                dists_sq = ox ** 2 + oz ** 2
                in_circle = dists_sq < (self.r_a ** 2)
                in_height = (oy >= step_clearance_m) & (oy <= head_clearance_m)
                colliding_mask = in_circle & in_height
                coll_count = int(np.sum(colliding_mask))
                return (coll_count >= self.obstacle_point_threshold), coll_count

            theta_rad = math.radians(theta_deg)
            sin_th = math.sin(theta_rad)
            cos_th = math.cos(theta_rad)

            s = -ox * sin_th - oz * cos_th
            h = ox * cos_th - oz * sin_th

            in_s = (s > 0.0) & (s < (rho_q + self.r_a))
            in_h = np.abs(h) < self.r_a

            d_horiz = D_raw if (D_raw is not None and D_raw > 1e-4) else max(rho_q, 1e-4)
            slope = p_y / d_horiz
            y_base_s = s * slope

            in_y = (oy >= (y_base_s + step_clearance_m)) & (oy <= (y_base_s + head_clearance_m))

            colliding_mask = in_s & in_h & in_y
            collision_count = int(np.sum(colliding_mask))
            return (collision_count >= self.obstacle_point_threshold), collision_count

        # 2D fallback mode
        if obstacles_2d.shape[0] == 0:
            return False, 0

        # If pure rotation (rho_q == 0), check in-place chassis footprint circle radius r_a
        if rho_q <= 0.0:
            dists_sq = obstacles_2d[:, 0] ** 2 + obstacles_2d[:, 1] ** 2
            in_circle = dists_sq < (self.r_a ** 2)
            coll_count = int(np.sum(in_circle))
            return (coll_count >= self.obstacle_point_threshold), coll_count

        theta_rad = math.radians(theta_deg)
        sin_th = math.sin(theta_rad)
        cos_th = math.cos(theta_rad)

        ox = obstacles_2d[:, 0]
        oz = obstacles_2d[:, 1]

        s = -ox * sin_th - oz * cos_th
        h = ox * cos_th - oz * sin_th

        in_s = (s > 0.0) & (s < (rho_q + self.r_a))
        in_h = np.abs(h) < self.r_a

        colliding_mask = in_s & in_h
        collision_count = int(np.sum(colliding_mask))
        return (collision_count >= self.obstacle_point_threshold), collision_count

    def evaluate(
        self,
        p_x: float,
        p_z: float,
        obstacles_2d: np.ndarray,
        m_eff: Optional[float] = None,
        p_y: float = 0.0,
        obstacles_3d: Optional[np.ndarray] = None,
    ) -> SafetyCheckResult:
        """Full safety check and motion planning pipeline for a candidate target (v7)."""
        if m_eff is None:
            m_eff = self.base_margin

        D_raw, theta_deg, rho_q, theta_q_deg = self.compute_kinematics(p_x, p_z, m_eff)

        # Handle zero displacement policy (Section 5.3 Item 5)
        if rho_q < self.forward_step_m:
            if theta_q_deg != 0.0 and self.zero_rho_policy == "rotate":
                # Execute turn-only move
                turn_steps = int(round(theta_q_deg / self.turn_step_deg))
                colliding, coll_count = self.test_corridor_collision(
                    theta_q_deg, 0.0, obstacles_2d, p_y=p_y, obstacles_3d=obstacles_3d, D_raw=D_raw
                )
                if colliding:
                    return SafetyCheckResult(
                        status="reselect",
                        cause="obstacle",
                        raw_distance_m=D_raw,
                        raw_bearing_deg=theta_deg,
                        effective_margin_m=m_eff,
                        quantized_rho_m=0.0,
                        quantized_theta_deg=theta_q_deg,
                        collision_point_count=coll_count,
                    )
                return SafetyCheckResult(
                    status="executed",
                    cause=None,
                    raw_distance_m=D_raw,
                    raw_bearing_deg=theta_deg,
                    effective_margin_m=m_eff,
                    quantized_rho_m=0.0,
                    quantized_theta_deg=theta_q_deg,
                    turn_steps=turn_steps,
                    forward_steps=0,
                )
            else:
                # Both are 0 or policy is reject -> cause "no_motion"
                return SafetyCheckResult(
                    status="reselect",
                    cause="no_motion",
                    raw_distance_m=D_raw,
                    raw_bearing_deg=theta_deg,
                    effective_margin_m=m_eff,
                    quantized_rho_m=0.0,
                    quantized_theta_deg=theta_q_deg,
                )

        # Test primary quantized corridor
        colliding, coll_count = self.test_corridor_collision(
            theta_q_deg, rho_q, obstacles_2d, p_y=p_y, obstacles_3d=obstacles_3d, D_raw=D_raw
        )

        snapped = False
        if colliding and self.allow_adjacent_bin_snap:
            delta_th = theta_deg - theta_q_deg
            if abs(delta_th) > 1e-4:
                alt_sign = 1.0 if delta_th > 0 else -1.0
                theta_alt_deg = theta_q_deg + alt_sign * self.turn_step_deg

                diff_alt = abs(theta_deg - theta_alt_deg)
                lateral_drift = rho_q * abs(math.sin(math.radians(diff_alt)))

                if (diff_alt <= self.max_snap_deg) and (lateral_drift <= self.max_snap_drift_m):
                    alt_colliding, alt_coll_count = self.test_corridor_collision(
                        theta_alt_deg, rho_q, obstacles_2d, p_y=p_y, obstacles_3d=obstacles_3d, D_raw=D_raw
                    )
                    if not alt_colliding:
                        theta_q_deg = theta_alt_deg
                        colliding = False
                        coll_count = 0
                        snapped = True

        if colliding:
            return SafetyCheckResult(
                status="reselect",
                cause="obstacle",
                raw_distance_m=D_raw,
                raw_bearing_deg=theta_deg,
                effective_margin_m=m_eff,
                quantized_rho_m=rho_q,
                quantized_theta_deg=theta_q_deg,
                collision_point_count=coll_count,
            )

        turn_steps = int(round(theta_q_deg / self.turn_step_deg))
        forward_steps = int(round(rho_q / self.forward_step_m))

        return SafetyCheckResult(
            status="executed",
            cause=None,
            raw_distance_m=D_raw,
            raw_bearing_deg=theta_deg,
            effective_margin_m=m_eff,
            quantized_rho_m=rho_q,
            quantized_theta_deg=theta_q_deg,
            turn_steps=turn_steps,
            forward_steps=forward_steps,
            collision_point_count=0,
            snapped_adjacent_bin=snapped,
        )
