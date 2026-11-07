"""
Rigorous 2D-to-3D Geometric Transformations & Depth Perception for Aerial VLN.
Handles camera intrinsics, resolution decoupling, patch median filtering,
and body-to-world kinematic propagation.
"""
import math
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class CameraIntrinsics:
    """Standard pinhole camera intrinsic parameters."""
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fov_deg: float = 90.0
    ):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        # Focal length for horizontal FOV: fx = (W / 2) / tan(FOV_rad / 2)
        fov_rad = math.radians(fov_deg)
        self.fx = (width / 2.0) / math.tan(fov_rad / 2.0)
        self.fy = self.fx  # Square pixels
        self.cx = width / 2.0
        self.cy = height / 2.0


def query_metric_depth_patch(
    depth_map: np.ndarray,
    u: float,
    v: float,
    patch_radius: int = 2,
    default_depth: float = 10.0,
    min_valid_depth: float = 0.1,
    max_valid_depth: float = 120.0
) -> float:
    """
    Robust metric depth query at normalized coordinates (u, v) in [0.0..1.0].
    Handles depth map resolution decoupling, out-of-bound clamping,
    and median filtering over a localized patch to suppress sensor noise/holes.
    """
    if depth_map is None or depth_map.size == 0:
        return default_depth

    dh, dw = depth_map.shape[:2]
    d_px = int(np.clip(round(u * (dw - 1)), 0, dw - 1))
    d_py = int(np.clip(round(v * (dh - 1)), 0, dh - 1))

    raw_val = float(depth_map[d_py, d_px])
    if min_valid_depth <= raw_val <= max_valid_depth and not (np.isnan(raw_val) or np.isinf(raw_val)):
        # Point query is valid, check patch for robustness
        y_min = max(0, d_py - patch_radius)
        y_max = min(dh, d_py + patch_radius + 1)
        x_min = max(0, d_px - patch_radius)
        x_max = min(dw, d_px + patch_radius + 1)
        patch = depth_map[y_min:y_max, x_min:x_max]
        valid = patch[(patch >= min_valid_depth) & (patch <= max_valid_depth) & (~np.isnan(patch)) & (~np.isinf(patch))]
        if valid.size > 0:
            return float(np.median(valid))
        return raw_val

    # If raw point was invalid/noisy, inspect neighboring patch
    y_min = max(0, d_py - patch_radius)
    y_max = min(dh, d_py + patch_radius + 1)
    x_min = max(0, d_px - patch_radius)
    x_max = min(dw, d_px + patch_radius + 1)
    patch = depth_map[y_min:y_max, x_min:x_max]
    valid = patch[(patch >= min_valid_depth) & (patch <= max_valid_depth) & (~np.isnan(patch)) & (~np.isinf(patch))]
    if valid.size > 0:
        return float(np.median(valid))

    return default_depth


def backproject_pixel_to_body_frame(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    max_range_m: float = 8.0,
    min_range_m: float = 1.5,
    stop_margin_m: float = 1.0,
    max_climb_rate_m: float = 3.0
) -> Tuple[float, float, float, float]:
    """
    Back-projects normalized (u, v) into continuous metric displacements
    in the UAV body coordinate system:
        X: Forward (+m)
        Y: Right (+m)
        Z: Up (+m)
        yaw_offset: Radians (+right / -left)
    """
    # 1. Normalized screen coordinates -> continuous camera pixel
    px = u * intrinsics.width
    py = v * intrinsics.height

    # 2. Camera Frame (Pinhole model):
    # X_cam: right (+), Y_cam: down (+), Z_cam: optical axis forward (+)
    x_cam = (px - intrinsics.cx) * depth_m / intrinsics.fx
    y_cam = (py - intrinsics.cy) * depth_m / intrinsics.fy
    z_cam = depth_m

    # 3. Camera Frame -> UAV Body Frame:
    # X_body (forward) = Z_cam
    # Y_body (right)   = X_cam
    # Z_body (up)      = -Y_cam
    x_body = z_cam
    y_body = x_cam
    z_body = -y_cam

    # 4. Range and bearing in horizontal plane
    range_xy = math.hypot(x_body, y_body)
    yaw_offset = math.atan2(y_body, x_body)

    # Clamped horizontal flight distance
    safe_target_range = max(0.0, range_xy - stop_margin_m)
    clamped_dist = min(max_range_m, max(min_range_m, safe_target_range))

    # Scale 3D vector along line-of-sight ray
    if range_xy > 1e-3:
        scale = clamped_dist / range_xy
        dx = x_body * scale
        dy = y_body * scale
        dz = np.clip(z_body * scale, -max_climb_rate_m, max_climb_rate_m)
    else:
        dx = clamped_dist
        dy = 0.0
        dz = 0.0

    return float(dx), float(dy), float(dz), float(yaw_offset)


def propagate_body_to_world(
    current_x: float,
    current_y: float,
    current_z: float,
    current_yaw_rad: float,
    dx_body: float,
    dy_body: float,
    dz_body: float,
    yaw_delta_rad: float
) -> Tuple[float, float, float, float]:
    """
    Propagates a body-frame displacement (dx, dy, dz, yaw_delta)
    into updated world coordinates (x, y, z, yaw) assuming standard right-hand rotation.
    """
    # Mid-heading integration for smooth continuous arcs
    mid_yaw = current_yaw_rad + (yaw_delta_rad * 0.5)
    cos_yaw = math.cos(mid_yaw)
    sin_yaw = math.sin(mid_yaw)

    # Rotate body (dx, dy) into world coordinates:
    # dx is forward, dy is right
    delta_world_x = dx_body * cos_yaw - dy_body * sin_yaw
    delta_world_y = dx_body * sin_yaw + dy_body * cos_yaw
    delta_world_z = dz_body

    next_x = current_x + delta_world_x
    next_y = current_y + delta_world_y
    next_z = current_z + delta_world_z
    next_yaw = normalize_angle_rad(current_yaw_rad + yaw_delta_rad)

    return next_x, next_y, next_z, next_yaw


def euclidean_distance_3d(p1, p2) -> float:
    """Computes 3D Euclidean distance between two points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    dz = p1[2] - p2[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def normalize_angle_rad(angle: float) -> float:
    """Normalizes an angle in radians to [-pi, pi]."""
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle

