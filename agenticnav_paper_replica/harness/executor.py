"""Tool execution layer for AgenticNav (v7 Specification)."""

import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from agenticnav_paper_replica.geometry.pinhole import (
    CameraConfig,
    deproject_pixel,
    camera_to_agent_frame,
    estimate_corridor_surface_slope,
    compute_effective_margin,
    extract_depth_patch,
    normalize_to_pixel,
    convert_raw_depth,
    parse_view_index,
    parse_view_index_ex,
    InvalidViewError,
    scan_upward_for_valid_depth,
    compute_azimuth_ground_projection,
)
from agenticnav_paper_replica.geometry.safety_region import (
    GroundCorridorChecker,
    SafetyCheckResult,
)
from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache, RecallResult


class ToolExecutor:
    """Dispatches and validates AgenticNav tool calls against active sensor frames."""

    def __init__(
        self,
        camera_config: CameraConfig,
        corridor_checker: GroundCorridorChecker,
        image_cache: EpisodicImageCache,
        standoff_mode: str = "fixed",  # "fixed" (default), "slope" (ablation)
        oracle_local_actuation: bool = False,
    ):
        self.config = camera_config
        self.checker = corridor_checker
        self.image_cache = image_cache
        self.standoff_mode = standoff_mode
        self.oracle_local_actuation = oracle_local_actuation
        self.blocked_view_count: int = 0

    def execute_query_depth(
        self,
        points: Union[List[Any], Dict[str, Any], str],
        depth_views: Dict[int, np.ndarray],
        obstacle_points_agent_xz: Optional[np.ndarray] = None,
        obstacle_points_agent_xyz: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute batched query_depth (Section 5.1).

        Returns center depth and 5x5 statistics in meters for each query point.
        """
        # Normalize points input
        if isinstance(points, str):
            try:
                points = json.loads(points)
            except Exception:
                points = []
        if isinstance(points, dict):
            points = [points]
        if not isinstance(points, list):
            points = []

        results = []
        for pt in points:
            if isinstance(pt, str):
                try:
                    pt = json.loads(pt)
                except Exception:
                    # Try regex extraction of view, u, v
                    import re
                    args = {}
                    vm = re.search(r'view[:=\s]+([0-3])', pt, re.I)
                    if vm: args["view"] = int(vm.group(1))
                    um = re.search(r'u[:=\s]+([0-9.]+)', pt, re.I)
                    if um: args["u"] = float(um.group(1))
                    vm2 = re.search(r'v[:=\s]+([0-9.]+)', pt, re.I)
                    if vm2: args["v"] = float(vm2.group(1))
                    pt = args if args else {}

            if not isinstance(pt, dict):
                continue

            raw_v = pt.get("view", 0)
            try:
                view, rewritten = parse_view_index_ex(raw_v)
            except InvalidViewError as e:
                logger.warning("Invalid view index in query_depth: %s", e)
                results.append({
                    "view": raw_v,
                    "u": round(float(pt.get("u", 0.5)), 2),
                    "v": round(float(pt.get("v", 0.5)), 2),
                    "error": "invalid_view",
                })
                continue

            u = float(pt.get("u", 0.5))
            v = float(pt.get("v", 0.5))

            if view not in depth_views:
                results.append({
                    "view": view,
                    "u": round(u, 2),
                    "v": round(v, 2),
                    "error": "invalid_depth",
                })
                continue

            depth_img = depth_views[view]

            # Check for blocked view due to excessive zero-depth pixels
            zero_fraction = float(np.mean(depth_img == 0.0))
            threshold = getattr(self.config, "blocked_view_zero_depth_threshold", 0.35)
            if zero_fraction >= threshold:
                results.append({
                    "view": view,
                    "u": round(u, 2),
                    "v": round(v, 2),
                    "error": "blocked_view",
                })
                continue
            x0, y0 = normalize_to_pixel(u, v, self.config)

            raw_val = float(depth_img[y0, x0])
            is_valid, z_center, error_cause = convert_raw_depth(raw_val, self.config)
            if not is_valid:
                if 0.0 < raw_val <= (self.config.d_min + 1e-3):
                    results.append({
                        "view": view,
                        "u": round(u, 2),
                        "v": round(v, 2),
                        "error": "near_plane_clip",
                        "message": (
                            "Target distance is < 0.50m (surface/wall/table in immediate foreground). "
                            "Look higher (v <= 0.50) or pick another direction."
                        ),
                    })
                elif raw_val >= (self.config.d_max - 1e-3):
                    results.append({
                        "view": view,
                        "u": round(u, 2),
                        "v": round(v, 2),
                        "error": "far_saturation",
                        "message": "Target distance is >= 5.0m (beyond sensor range).",
                    })
                else:
                    results.append({
                        "view": view,
                        "u": round(u, 2),
                        "v": round(v, 2),
                        "error": error_cause or "invalid_depth",
                    })
                continue

            patch = extract_depth_patch(depth_img, x0, y0, patch_size=5)
            valid_patch_depths = []
            for r in range(5):
                for c in range(5):
                    pv = float(patch[r, c])
                    ok, d_met, _ = convert_raw_depth(pv, self.config)
                    if ok:
                        valid_patch_depths.append(d_met)

            # Compute action preview (theta, rho, clearance) that move_to would execute
            p_c_prev = deproject_pixel(x0, y0, z_center, self.config, is_normalized=False)
            p_ag_prev = camera_to_agent_frame(p_c_prev, view, self.config)
            px = float(p_ag_prev[0])
            py = float(p_ag_prev[1])
            pz = float(p_ag_prev[2])

            if obstacle_points_agent_xz is not None and not self.oracle_local_actuation:
                safety_res = self.checker.evaluate(
                    p_x=px,
                    p_z=pz,
                    obstacles_2d=obstacle_points_agent_xz,
                    m_eff=self.checker.base_margin,
                    p_y=py,
                    obstacles_3d=obstacle_points_agent_xyz,
                )
                is_clear = (safety_res.status == "executed")
                obstacle_cause = safety_res.cause if not is_clear else None
                raw_rho = safety_res.quantized_rho_m
                th_q_prev = safety_res.quantized_theta_deg
            else:
                _, _, raw_rho, th_q_prev = self.checker.compute_kinematics(px, pz, self.checker.base_margin)
                is_clear = True
                obstacle_cause = None

            # Unified terminal stride scaling with move_to:
            # Damp v >= 0.80 to 1.0m only during near-field approach (depth < 2.5m)
            is_terminal_near_field = (v >= 0.80 and z_center < 2.5)
            if is_clear:
                if is_terminal_near_field and raw_rho > 1.0:
                    exec_rho = 1.0
                else:
                    exec_rho = raw_rho
            else:
                exec_rho = 0.0

            if len(valid_patch_depths) >= 3:
                min_m = float(np.min(valid_patch_depths))
                mean_m = float(np.mean(valid_patch_depths))
                median_m = float(np.median(valid_patch_depths))
                results.append({
                    "view": view,
                    "u": round(u, 2),
                    "v": round(v, 2),
                    "depth_m": round(z_center, 2),
                    "min_m": round(min_m, 2),
                    "mean_m": round(mean_m, 2),
                    "median_m": round(median_m, 2),
                    "preview_theta_deg": round(th_q_prev, 1),
                    "preview_rho_m": round(exec_rho, 2),
                    "clear": is_clear,
                    "obstacle_cause": obstacle_cause,
                })
            else:
                results.append({
                    "view": view,
                    "u": round(u, 2),
                    "v": round(v, 2),
                    "depth_m": round(z_center, 2),
                    "min_m": None,
                    "mean_m": None,
                    "median_m": None,
                    "sparse_patch": True,
                    "preview_theta_deg": round(th_q_prev, 1),
                    "preview_rho_m": round(exec_rho, 2),
                    "clear": is_clear,
                    "obstacle_cause": obstacle_cause,
                })

        return {"query_results": results}

    def execute_recall(
        self,
        node: int,
        view: int,
    ) -> Dict[str, Any]:
        """Execute recall tool (Section 5.2)."""
        recall_res: RecallResult = self.image_cache.recall(node, view)
        if not recall_res.success:
            return {
                "status": "error",
                "message": recall_res.error_message,
                "node": node,
                "view": view,
            }

        return {
            "status": "success",
            "node": node,
            "view": view,
            "image": recall_res.image,
        }

    def execute_move_to(
        self,
        view: int,
        u: float,
        v: float,
        reason: str,
        depth_views: Dict[int, np.ndarray],
        obstacle_points_agent_xz: np.ndarray,
        obstacle_points_agent_xyz: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Execute move_to tool (Section 5.3)."""
        if view not in depth_views:
            return {
                "status": "reselect",
                "cause": "invalid_depth",
            }

        depth_img = depth_views[view]

        # Check for blocked view due to excessive zero-depth pixels (unrendered / near-plane clipping)
        zero_fraction = float(np.mean(depth_img == 0.0))
        threshold = getattr(self.config, "blocked_view_zero_depth_threshold", 0.35)
        if zero_fraction >= threshold:
            self.blocked_view_count += 1
            import logging
            logging.getLogger(__name__).warning(
                f"View {view} blocked: zero-depth pixel fraction {zero_fraction:.4f} >= {threshold:.4f} "
                f"(total blocked view events: {self.blocked_view_count})"
            )
            return {
                "status": "reselect",
                "cause": "blocked_view",
                "zero_depth_fraction": round(zero_fraction, 4),
            }
        x0, y0 = normalize_to_pixel(u, v, self.config)

        raw_val = float(depth_img[y0, x0])
        is_valid, z_c, error_cause = convert_raw_depth(raw_val, self.config)
        used_fallback = False

        if not is_valid:
            if 0.0 < raw_val <= (self.config.d_min + 1e-3):
                # 1. Try upward vertical ray search along column u
                scan_res = scan_upward_for_valid_depth(
                    depth_img=depth_img,
                    u=u,
                    v=v,
                    config=self.config,
                    min_v=0.35,
                    step_v=0.04,
                )
                if scan_res is not None:
                    v_cand, x0_cand, y0_cand, z_c_cand = scan_res
                    x0, y0 = x0_cand, y0_cand
                    v = v_cand
                    z_c = z_c_cand
                    is_valid = True
                elif self.oracle_local_actuation:
                    # 2. Azimuth ground-projection fallback for oracle actuation
                    used_fallback = True
                    z_c = 1.25
                    is_valid = True

            if not is_valid:
                if 0.0 < raw_val <= (self.config.d_min + 1e-3):
                    return {
                        "status": "reselect",
                        "cause": "invalid_depth",
                        "detail": "near_plane_clip",
                        "message": (
                            "NEAR_PLANE_CLIP (d < 0.50m): You are standing < 0.50m from a surface/wall/table in this view. "
                            "(1) If you have already reached the goal location, call stop() now! "
                            "(2) Do NOT make tiny adjustments to (u, v) around the same spot; either look higher (v in [0.35, 0.50]) "
                            "or pick an open direction in another view (View 1, 2, or 3)."
                        ),
                    }
                elif raw_val >= (self.config.d_max - 1e-3):
                    return {
                        "status": "reselect",
                        "cause": "invalid_depth",
                        "detail": "far_saturation",
                        "message": "FAR_SATURATION (d >= 5.0m): Target is beyond camera range (5.0m). Pick a closer target (v in [0.60, 0.85]).",
                    }
                else:
                    return {
                        "status": "reselect",
                        "cause": error_cause or "invalid_depth",
                    }

        # Step 2: Deproject to camera optical vector and canonical agent frame
        if used_fallback:
            p_agent = compute_azimuth_ground_projection(
                view_idx=view,
                u=u,
                config=self.config,
                distance_m=1.25,
            )
            p_x, p_y, p_z = float(p_agent[0]), float(p_agent[1]), float(p_agent[2])
            rho_raw = math.sqrt(p_x * p_x + p_z * p_z)
            rho_executed = min(rho_raw, self.checker.rho_max)
            fwd_steps = int(round(rho_executed / self.checker.forward_step_m))
            th_raw = math.degrees(math.atan2(-p_x, -p_z))
            th_deg = self.checker.turn_step_deg * round(th_raw / self.checker.turn_step_deg)
            turn_steps = int(round(th_deg / self.checker.turn_step_deg))

            return {
                "status": "executed",
                "theta_deg": round(th_deg, 1),
                "rho_m": round(rho_executed, 2),
                "moved_m": round(rho_executed, 2),
                "collided": False,
                "turn_steps": turn_steps,
                "forward_steps": fwd_steps,
                "p_agent": [round(p_x, 3), round(p_y, 3), round(p_z, 3)],
                "view": view,
                "u": u,
                "v": v,
                "reason": reason,
                "fallback": "azimuth_ground_projection",
            }

        p_c = deproject_pixel(x0, y0, z_c, self.config, is_normalized=False)
        p_agent = camera_to_agent_frame(p_c, view, self.config)
        p_x, p_y, p_z = float(p_agent[0]), float(p_agent[1]), float(p_agent[2])

        # Step 3: Margin mode (fixed by default, slope in corridor frame if ablation on)
        if self.standoff_mode == "slope":
            patch = extract_depth_patch(depth_img, x0, y0, patch_size=5)
            raw_theta_deg = math.degrees(math.atan2(-p_x, -p_z))
            theta_q_init = 15.0 * round(raw_theta_deg / 15.0)
            tan_alpha = estimate_corridor_surface_slope(
                patch, view, x0, y0, theta_q_init, self.config
            )
            m_eff = compute_effective_margin(
                tan_alpha,
                base_margin=self.checker.base_margin,
                robot_radius=self.checker.r_a,
                standoff_mode="slope",
            )
        else:
            m_eff = self.checker.base_margin

        # Step 4, 5, 6: Corridor check and quantization (with 3D swept cylinder evaluation)
        safety_res: SafetyCheckResult = self.checker.evaluate(
            p_x=p_x,
            p_z=p_z,
            obstacles_2d=obstacle_points_agent_xz,
            m_eff=m_eff,
            p_y=p_y,
            obstacles_3d=obstacle_points_agent_xyz,
        )

        if safety_res.status == "executed" or self.oracle_local_actuation:
            if safety_res.status == "executed":
                rho_executed = safety_res.quantized_rho_m
                fwd_steps = safety_res.forward_steps
                th_deg = safety_res.quantized_theta_deg
                turn_steps = safety_res.turn_steps
            else:
                # Oracle actuation: NavMesh path follower handles obstacles
                rho_raw = math.sqrt(p_x * p_x + p_z * p_z)
                rho_executed = min(rho_raw, self.checker.rho_max)
                fwd_steps = int(round(rho_executed / self.checker.forward_step_m))
                th_raw = math.degrees(math.atan2(-p_x, -p_z))
                th_deg = self.checker.turn_step_deg * round(th_raw / self.checker.turn_step_deg)
                turn_steps = int(round(th_deg / self.checker.turn_step_deg))

            # Near-field terminal stride scaling (prevents momentum overshoot past 3.0m goal circle)
            reason_lower = str(reason).lower()
            terminal_indicators = (
                "stop at", "stop near", "stop in", "stop by", "stop next", "stop front",
                "final landmark", "target landmark", "destination landmark", "stopping landmark",
                "approaching landmark", "approaching the target", "arrived at"
            )
            is_terminal_intent = (
                any(ti in reason_lower for ti in terminal_indicators)
                or (v >= 0.80 and z_c < 2.5)
            )
            if is_terminal_intent and rho_executed > 1.0:
                rho_executed = 1.0
                fwd_steps = int(round(rho_executed / self.checker.forward_step_m))

            return {
                "status": "executed",
                "theta_deg": round(th_deg, 1),
                "rho_m": round(rho_executed, 2),
                "moved_m": round(rho_executed, 2),
                "collided": False,
                "turn_steps": turn_steps,
                "forward_steps": fwd_steps,
                "p_agent": [round(p_x, 3), round(p_y, 3), round(p_z, 3)],
                "view": view,
                "u": u,
                "v": v,
                "reason": reason,
            }
        else:
            return {
                "status": "reselect",
                "cause": safety_res.cause,
            }

    def execute_stop(self, reason: str) -> Dict[str, Any]:
        """Execute stop tool (Section 5.4)."""
        return {
            "status": "stopped",
            "reason": reason,
        }
