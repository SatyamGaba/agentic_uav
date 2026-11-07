"""Pinhole camera geometry and coordinate transformations (v7 Specification).

Canonical Right-Handed Agent Frame (matching Habitat-Sim):
    - Forward:  -Z_agent
    - Right:    +X_agent
    - Up:       +Y_agent (Floor datum y=0, Camera center at y = h_cam = 1.25m)

Camera Optical Frame:
    - Forward:  +z (optical depth z_c)
    - Right:    +x
    - Down:     +y

View Azimuths (CCW-positive around +Y_agent):
    - View 0: Front  (psi_0 =   0 deg)
    - View 1: Right  (psi_1 = -90 deg)
    - View 2: Back   (psi_2 = 180 deg)
    - View 3: Left   (psi_3 = +90 deg)
"""

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


VIEW_AZIMUTHS_DEG: Dict[int, float] = {
    0: 0.0,       # Front
    1: -90.0,     # Right
    2: 180.0,     # Back
    3: 90.0,      # Left
}

VIEW_LABELS: Dict[int, str] = {
    0: "Front (0°)",
    1: "Right (-90°)",
    2: "Back (180°)",
    3: "Left (+90°)",
}


def parse_view_index(val: Any) -> int:
    """Robustly parse a view index from an int, float, or string (e.g. 0, 'Front', 'View 0', 'Right')."""
    if val is None:
        return 0
    if isinstance(val, (int, np.integer)):
        return int(val) % 4
    if isinstance(val, float):
        return int(round(val)) % 4
    s = str(val).strip().lower()
    if "front" in s or "forward" in s:
        return 0
    if "right" in s:
        return 1
    if "back" in s or "behind" in s or "rear" in s:
        return 2
    if "left" in s:
        return 3
    for char in s:
        if char in "0123":
            return int(char)
    return 0


@dataclass
class CameraConfig:
    """Sensor parameters matching standard R2R-CE VLN-CE Habitat-Sim setup."""
    image_width: int = 640
    image_height: int = 480
    hfov_deg: float = 90.0
    d_min: float = 0.5   # meters: waypoint query floor (paper sensor specification)
    d_max: float = 5.0   # meters: sensor max range
    min_obstacle_depth: float = 0.05  # meters: near obstacle detection threshold (deviation from 0.5m sensor floor)
    camera_height: float = 1.25  # meters above floor (y=0)
    is_raw_normalized: bool = False  # False for metric buffers in meters; True if [0, 1] raw

    @property
    def hfov_rad(self) -> float:
        return math.radians(self.hfov_deg)

    @property
    def fx(self) -> float:
        return (self.image_width / 2.0) / math.tan(self.hfov_rad / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.image_width / 2.0

    @property
    def cy(self) -> float:
        return self.image_height / 2.0


def normalize_to_pixel(u: float, v: float, config: CameraConfig) -> Tuple[int, int]:
    """Convert normalized (u, v) in [0, 1] to 0-indexed integer pixel (x0, y0) using floor-clamping.

    Formula from Blueprint v7 Section 5.1:
        x0 = clip(floor(u * W), 0, W - 1)
        y0 = clip(floor(v * H), 0, H - 1)
    """
    x0 = int(np.clip(math.floor(float(u) * config.image_width), 0, config.image_width - 1))
    y0 = int(np.clip(math.floor(float(v) * config.image_height), 0, config.image_height - 1))
    return x0, y0


def pixel_to_normalized(x0: int, y0: int, config: CameraConfig) -> Tuple[float, float]:
    """Convert integer pixel (x0, y0) to normalized (u, v) in [0, 1]."""
    u = x0 / (config.image_width - 1)
    v = y0 / (config.image_height - 1)
    return float(u), float(v)


def convert_raw_depth(
    val: float,
    config: CameraConfig,
    is_raw_normalized: Optional[bool] = None,
    eps: float = 1e-3,
) -> Tuple[bool, float, Optional[str]]:
    """Convert depth value to metric depth in meters.

    Rejects invalid depth: NaN, Inf, non-positive, near-clipping clamp (<= d_min + eps),
    and far saturation (>= d_max - eps).

    Returns:
        (is_valid, depth_m, error_cause)
    """
    if val is None or np.isnan(val) or np.isinf(val) or val <= 0.0:
        return False, 0.0, "invalid_depth"

    use_norm = config.is_raw_normalized if is_raw_normalized is None else is_raw_normalized

    if use_norm:
        if val <= 0.0 or val >= 1.0:
            return False, 0.0, "invalid_depth"
        d_metric = config.d_min + val * (config.d_max - config.d_min)
        return True, float(d_metric), None
    else:
        # Metric depth in meters: mask near clamp (<= d_min + eps) and far saturation as invalid
        if val <= (config.d_min + eps) or val >= (config.d_max - eps):
            return False, float(val), "invalid_depth"
        return True, float(val), None


def deproject_pixel(
    u_or_x0: Union[int, float],
    v_or_y0: Union[int, float],
    z_c: float,
    config: CameraConfig,
    is_normalized: bool = True,
) -> np.ndarray:
    """Deproject image coordinates and planar metric depth z_c to optical 3D vector [x_c, y_c, z_c].

    Uses the standard pinhole inverse projection with half-pixel centering (+0.5):
        x_c = (x0 + 0.5 - cx) * z_c / fx
        y_c = (y0 + 0.5 - cy) * z_c / fy
        z_c = z_c
    """
    if is_normalized:
        x0, y0 = normalize_to_pixel(float(u_or_x0), float(v_or_y0), config)
    else:
        x0, y0 = int(u_or_x0), int(v_or_y0)

    x_c = (x0 + 0.5 - config.cx) * z_c / config.fx
    y_c = (y0 + 0.5 - config.cy) * z_c / config.fy
    return np.array([x_c, y_c, float(z_c)], dtype=np.float64)


def reproject_point(p_c: np.ndarray, config: CameraConfig) -> Tuple[float, float, float]:
    """Reproject 3D optical point [x_c, y_c, z_c] back to image pixel coordinates (x, y) and depth z_c."""
    x_c, y_c, z_c = float(p_c[0]), float(p_c[1]), float(p_c[2])
    x0 = x_c * config.fx / z_c + config.cx - 0.5
    y0 = y_c * config.fy / z_c + config.cy - 0.5
    return x0, y0, z_c


def camera_to_agent_frame(
    p_c: np.ndarray,
    view_idx: int,
    config: CameraConfig,
) -> np.ndarray:
    """Transform camera optical coordinates [x_c, y_c, z_c] to canonical agent frame [p_x, p_y, p_z]."""
    if view_idx not in VIEW_AZIMUTHS_DEG:
        raise ValueError(f"Invalid view_idx={view_idx}. Must be in {list(VIEW_AZIMUTHS_DEG.keys())}.")

    psi_rad = math.radians(VIEW_AZIMUTHS_DEG[view_idx])
    cos_psi = math.cos(psi_rad)
    sin_psi = math.sin(psi_rad)

    x_c = float(p_c[0])
    y_c = float(p_c[1])
    z_c = float(p_c[2])

    x_v = x_c
    y_v = -y_c + config.camera_height
    z_v = -z_c

    p_x = cos_psi * x_v + sin_psi * z_v
    p_y = y_v
    p_z = -sin_psi * x_v + cos_psi * z_v

    return np.array([p_x, p_y, p_z], dtype=np.float64)


def agent_to_camera_frame(
    p_agent: np.ndarray,
    view_idx: int,
    config: CameraConfig,
) -> np.ndarray:
    """Transform canonical agent coordinates [p_x, p_y, p_z] back to camera optical coordinates [x_c, y_c, z_c].

    Exact mathematical inverse of camera_to_agent_frame.
    """
    if view_idx not in VIEW_AZIMUTHS_DEG:
        raise ValueError(f"Invalid view_idx={view_idx}. Must be in {list(VIEW_AZIMUTHS_DEG.keys())}.")

    psi_rad = math.radians(VIEW_AZIMUTHS_DEG[view_idx])
    cos_psi = math.cos(psi_rad)
    sin_psi = math.sin(psi_rad)

    p_x = float(p_agent[0])
    p_y = float(p_agent[1])
    p_z = float(p_agent[2])

    x_v = cos_psi * p_x - sin_psi * p_z
    y_v = p_y
    z_v = sin_psi * p_x + cos_psi * p_z

    x_c = x_v
    y_c = config.camera_height - y_v
    z_c = -z_v

    return np.array([x_c, y_c, z_c], dtype=np.float64)


def extract_depth_patch(
    depth_image: np.ndarray,
    x0: int,
    y0: int,
    patch_size: int = 5,
) -> np.ndarray:
    """Extract a patch_size x patch_size depth patch centered at (x0, y0) with boundary clamping."""
    h, w = depth_image.shape
    half = patch_size // 2
    r_min = max(0, y0 - half)
    r_max = min(h, y0 + half + 1)
    c_min = max(0, x0 - half)
    c_max = min(w, x0 + half + 1)

    patch = np.full((patch_size, patch_size), np.nan, dtype=np.float32)
    sub = depth_image[r_min:r_max, c_min:c_max]

    out_r_min = half - (y0 - r_min)
    out_r_max = out_r_min + (r_max - r_min)
    out_c_min = half - (x0 - c_min)
    out_c_max = out_c_min + (c_max - c_min)

    patch[out_r_min:out_r_max, out_c_min:out_c_max] = sub
    return patch


def estimate_corridor_surface_slope(
    depth_patch: np.ndarray,
    view_idx: int,
    x0: int,
    y0: int,
    theta_q_deg: float,
    config: CameraConfig,
    inlier_depth_threshold: float = 0.30,
    max_slope_deg: float = 60.0,
) -> float:
    """Estimate local surface slope tan(alpha) in the corridor frame (Section 11 ablation)."""
    if depth_patch.shape != (5, 5):
        return 0.0

    d_center = float(depth_patch[2, 2])
    if not (config.d_min <= d_center <= config.d_max) or np.isnan(d_center):
        return 0.0

    theta_rad = math.radians(theta_q_deg)
    b_x = -math.sin(theta_rad)
    b_z = -math.cos(theta_rad)
    l_x = math.cos(theta_rad)
    l_z = -math.sin(theta_rad)

    s_coords = []
    h_coords = []

    for r in range(5):
        for c in range(5):
            z_val = float(depth_patch[r, c])
            if np.isnan(z_val) or not (config.d_min <= z_val <= config.d_max):
                continue
            if abs(z_val - d_center) > inlier_depth_threshold:
                continue

            px_col = x0 + (c - 2)
            px_row = y0 + (r - 2)
            p_c = deproject_pixel(px_col, px_row, z_val, config, is_normalized=False)
            p_agent = camera_to_agent_frame(p_c, view_idx, config)

            qx = float(p_agent[0])
            qz = float(p_agent[2])
            s = qx * b_x + qz * b_z
            h = qx * l_x + qz * l_z

            s_coords.append(s)
            h_coords.append(h)

    if len(h_coords) < 5:
        return 0.0

    h_arr = np.array(h_coords, dtype=np.float64)
    s_arr = np.array(s_coords, dtype=np.float64)

    var_h = float(np.var(h_arr))
    if var_h < 1e-6:
        return 0.0

    cov_hs = float(np.mean((h_arr - np.mean(h_arr)) * (s_arr - np.mean(s_arr))))
    slope_c = cov_hs / var_h

    max_slope_tan = math.tan(math.radians(max_slope_deg))
    return float(np.clip(abs(slope_c), 0.0, max_slope_tan))


def compute_effective_margin(
    tan_alpha: float = 0.0,
    base_margin: float = 0.30,
    robot_radius: float = 0.18,
    standoff_mode: str = "fixed",
) -> float:
    """Compute effective standoff margin m_eff (Section 11)."""
    if standoff_mode == "fixed":
        return float(base_margin)
    else:
        margin_slope = robot_radius * (1.0 + tan_alpha) + 0.05
        return float(max(base_margin, margin_slope))


def unproject_depth_image(
    depth_image: np.ndarray,
    view_idx: int,
    config: CameraConfig,
    stride: int = 4,
    y_min: float = 0.10,
    y_max: float = 1.25,
    eps: float = 1e-3,
) -> np.ndarray:
    """Deproject full 2D depth image into 3D obstacle pointcloud in canonical agent frame.

    Masks near-clipping clamp (<= d_min + eps) and far-saturation (>= d_max - eps).
    """
    h, w = depth_image.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)

    z_vals = depth_image[grid_y, grid_x].astype(np.float64)

    min_obs_d = getattr(config, "min_obstacle_depth", 0.05)
    valid_mask = (
        (z_vals >= min_obs_d)
        & (z_vals <= (config.d_max - eps))
        & (~np.isnan(z_vals))
        & (~np.isinf(z_vals))
    )
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float64)

    z_valid = z_vals[valid_mask]
    gx_valid = grid_x[valid_mask]
    gy_valid = grid_y[valid_mask]

    x_c = (gx_valid + 0.5 - config.cx) * z_valid / config.fx
    y_c = (gy_valid + 0.5 - config.cy) * z_valid / config.fy
    z_c = z_valid

    x_v = x_c
    y_v = -y_c + config.camera_height
    z_v = -z_c

    height_mask = (y_v >= y_min) & (y_v <= y_max)
    if not np.any(height_mask):
        return np.empty((0, 3), dtype=np.float64)

    x_v = x_v[height_mask]
    y_v = y_v[height_mask]
    z_v = z_v[height_mask]

    psi_rad = math.radians(VIEW_AZIMUTHS_DEG[view_idx])
    cos_psi = math.cos(psi_rad)
    sin_psi = math.sin(psi_rad)

    p_x = cos_psi * x_v + sin_psi * z_v
    p_y = y_v
    p_z = -sin_psi * x_v + cos_psi * z_v

    return np.column_stack([p_x, p_y, p_z])


def extract_view_obstacle_and_free_points(
    depth_image: np.ndarray,
    view_idx: int,
    config: CameraConfig,
    stride: int = 4,
    y_min: float = 0.10,
    y_max: float = 1.25,
    eps: float = 1e-3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Deproject depth image into both 3D chassis obstacles and 2D free-space ray endpoints.

    Rejects invalid depth: near-clipping clamp (<= d_min + eps), NaN, Inf, non-positive.
    Classifies far-saturation (>= d_max - eps) as free-space endpoints, not chassis obstacles.

    Returns:
        (obstacle_pts_3d, free_endpoints_2d):
            - obstacle_pts_3d: (N, 3) obstacle points inside chassis height band [y_min, y_max].
            - free_endpoints_2d: (M, 2) [p_x, p_z] ground endpoints for floor hits and max-range rays.
    """
    h, w = depth_image.shape
    ys = np.arange(0, h, stride)
    xs = np.arange(0, w, stride)
    grid_x, grid_y = np.meshgrid(xs, ys)

    z_vals = depth_image[grid_y, grid_x].astype(np.float64)
    gx_flat = grid_x.flatten()
    gy_flat = grid_y.flatten()
    z_flat = z_vals.flatten()

    min_obs_d = getattr(config, "min_obstacle_depth", 0.05)
    # Valid depth: not NaN, not Inf, and >= min_obs_d - eps
    valid_mask = (~np.isnan(z_flat)) & (~np.isinf(z_flat)) & (z_flat >= (min_obs_d - eps))
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    gx_v = gx_flat[valid_mask]
    gy_v = gy_flat[valid_mask]
    z_v_raw = z_flat[valid_mask]

    # Clip to [min_obs_d, d_max] for projection
    z_c = np.clip(z_v_raw, min_obs_d, config.d_max)
    x_c = (gx_v + 0.5 - config.cx) * z_c / config.fx
    y_c = (gy_v + 0.5 - config.cy) * z_c / config.fy

    x_v = x_c
    y_v = -y_c + config.camera_height
    z_v = -z_c

    psi_rad = math.radians(VIEW_AZIMUTHS_DEG[view_idx])
    cos_psi = math.cos(psi_rad)
    sin_psi = math.sin(psi_rad)

    p_x = cos_psi * x_v + sin_psi * z_v
    p_y = y_v
    p_z = -sin_psi * x_v + cos_psi * z_v

    # Classify:
    # 1. Saturated max range (>= d_max - eps): free ray endpoint
    is_max_range = (z_v_raw >= (config.d_max - eps))
    # 2. Obstacles: inside chassis height band and within valid sensor linear range
    is_obstacle = (~is_max_range) & (p_y >= y_min) & (p_y <= y_max)
    # 3. Floor hits: below chassis height and not saturated
    is_floor = (~is_max_range) & (p_y < y_min)

    obs_pts = np.column_stack([p_x[is_obstacle], p_y[is_obstacle], p_z[is_obstacle]])

    is_free_endpoint = is_max_range | is_floor
    free_endpoints = np.column_stack([p_x[is_free_endpoint], p_z[is_free_endpoint]])

    return obs_pts, free_endpoints

