"""
3D Metric Scene Graph
=====================
Maintains an environment-agnostic, zero-prior 3D object scene graph from
RGB-D backprojection and SLAM map points.

Zero hardcoded volume dictionaries: physical centroids and dimensions
are measured directly from point cloud distributions.
"""

from __future__ import annotations
import math
import difflib
import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from contracts import SceneNode, DroneTelemetry

logger = logging.getLogger("MavicVLM")


def compute_camera_to_world_matrix(telemetry: DroneTelemetry) -> np.ndarray:
    """Computes 4x4 SE(3) transformation matrix T_wc (Camera RDF -> World ENU).
    
    Transforms points from Camera optical frame [X_right, Y_down, Z_forward]
    into World ENU [X_east, Y_north, Z_up].
    """
    yaw = telemetry.yaw
    gimbal_pitch = telemetry.gimbal_pitch

    # Base Camera RDF to Body FLU rotation (at 0 gimbal tilt):
    # X_b (forward) = +Z_c
    # Y_b (left)    = -X_c
    # Z_b (up)      = -Y_c
    R_c_to_b0 = np.array([
        [ 0.0,  0.0,  1.0],
        [-1.0,  0.0,  0.0],
        [ 0.0, -1.0,  0.0],
    ], dtype=np.float64)

    # Gimbal pitch rotation around body Y_b (pitching down rotates around -Y_b):
    cos_p, sin_p = math.cos(gimbal_pitch), math.sin(gimbal_pitch)
    R_pitch = np.array([
        [ cos_p,  0.0, -sin_p],
        [   0.0,  1.0,    0.0],
        [ sin_p,  0.0,  cos_p],
    ], dtype=np.float64)

    # Airframe Yaw rotation in world ENU:
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    R_yaw = np.array([
        [cos_y, -sin_y, 0.0],
        [sin_y,  cos_y, 0.0],
        [  0.0,    0.0, 1.0],
    ], dtype=np.float64)

    # Combined rotation: World <- Body <- Pitch <- Camera
    R_wc = R_yaw @ R_pitch @ R_c_to_b0

    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_wc
    T_wc[:3, 3] = telemetry.position
    return T_wc


def backproject_bbox(
    bbox_norm: Tuple[float, float, float, float],
    depth_map: np.ndarray,
    K: np.ndarray,
    T_wc: np.ndarray,
    valid_range: Tuple[float, float] = (0.3, 120.0),  # Calibrated from AirSim urban level scale (67m-100m structures); rejects sky sentinels (>500m)
    max_samples: int = 1500,
) -> np.ndarray:
    """Backprojects 2D bounding box pixels with depth into 3D World ENU coordinates.
    
    Args:
        bbox_norm: [u_min, v_min, u_max, v_max] normalized in [0, 1]
        depth_map: HxW float32 metric depth array in meters
        K: 3x3 camera matrix
        T_wc: 4x4 SE(3) Camera RDF -> World ENU transform
        valid_range: (min_depth, max_depth) in meters (default 0.3-120.0m calibrated for city block scales)
        max_samples: max points to sample to keep compute bounded
        
    Returns:
        (N, 3) numpy array of 3D points in World ENU
    """
    h, w = depth_map.shape[:2]
    u1 = max(0, int(bbox_norm[0] * w))
    v1 = max(0, int(bbox_norm[1] * h))
    u2 = min(w, int(bbox_norm[2] * w))
    v2 = min(h, int(bbox_norm[3] * h))

    if u2 <= u1 or v2 <= v1:
        return np.empty((0, 3), dtype=np.float64)

    bbox_depth = depth_map[v1:v2, u1:u2]
    v_coords, u_coords = np.nonzero((bbox_depth >= valid_range[0]) & (bbox_depth <= valid_range[1]))
    if len(u_coords) == 0:
        return np.empty((0, 3), dtype=np.float64)

    # Deterministic spatial uniform striding if too dense
    if len(u_coords) > max_samples:
        step = max(1, len(u_coords) // max_samples)
        u_coords = u_coords[::step][:max_samples]
        v_coords = v_coords[::step][:max_samples]

    actual_u = u1 + u_coords
    actual_v = v1 + v_coords
    actual_d = bbox_depth[v_coords, u_coords]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Camera optical coordinates (RDF)
    x_c = (actual_u - cx) * actual_d / fx
    y_c = (actual_v - cy) * actual_d / fy
    z_c = actual_d

    pts_c = np.vstack([x_c, y_c, z_c, np.ones_like(z_c)]) # 4xN
    pts_w = T_wc @ pts_c # 4xN
    return pts_w[:3, :].T # Nx3


def robust_centroid_and_extent(points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes median centroid, quantile physical extent, and 2D orientation."""
    if len(points_world) == 0:
        return np.zeros(3), np.zeros(3), 0.0

    # 1. Median centroid (robust against foreground/background outliers)
    centroid = np.median(points_world, axis=0)

    # 2. 5th to 95th quantile spans along world axes
    q05 = np.percentile(points_world, 5, axis=0)
    q95 = np.percentile(points_world, 95, axis=0)
    dimensions = np.maximum(0.3, q95 - q05)

    # 3. 2D PCA orientation of ground footprint
    xy_centered = points_world[:, :2] - centroid[:2]
    cov = np.cov(xy_centered, rowvar=False)
    if cov.shape == (2, 2) and not np.isnan(cov).any():
        evals, evecs = np.linalg.eigh(cov)
        principal_axis = evecs[:, 1]
        yaw_box = float(math.atan2(principal_axis[1], principal_axis[0]))
    else:
        yaw_box = 0.0

    return centroid, dimensions, yaw_box


def is_semantic_match(label_a: str, label_b: str, threshold: float = 0.65) -> bool:
    """Semantic match check using substring inclusion and normalized SequenceMatcher."""
    a, b = label_a.lower().strip(), label_b.lower().strip()
    if a == b or a in b or b in a:
        return True
    ratio = difflib.SequenceMatcher(None, a, b).ratio()
    return ratio >= threshold


class MetricSceneGraph3D:
    """Maintains active 3D grounded objects with zero hardcoded volumetric dictionaries."""

    def __init__(
        self,
        match_radius_base: float = 2.5,
        min_views_to_confirm: int = 2,
        min_viewpoint_baseline_m: float = 1.5,
        max_nodes: int = 200,
    ):
        self.nodes: Dict[int, SceneNode] = {}
        self.next_node_id = 1
        self.match_radius_base = match_radius_base
        self.min_views_to_confirm = min_views_to_confirm
        self.min_viewpoint_baseline_m = min_viewpoint_baseline_m
        self.max_nodes = max_nodes

    def add_or_update_detection(
        self,
        label: str,
        bbox_norm: Tuple[float, float, float, float],
        depth_map: np.ndarray,
        K: np.ndarray,
        T_wc: np.ndarray,
        uav_position: np.ndarray,
        timestamp: float,
        detection_confidence: float = 0.8,
        slam_points: Optional[List[np.ndarray]] = None,
    ) -> Optional[SceneNode]:
        """Backprojects detection, fuses with existing nodes or instantiates candidate."""
        pts_w = backproject_bbox(bbox_norm, depth_map, K, T_wc)
        
        # Optionally incorporate triangulated SLAM map points
        if slam_points is not None and len(slam_points) >= 3:
            pts_w = np.vstack([pts_w, np.array(slam_points)]) if len(pts_w) > 0 else np.array(slam_points)

        if len(pts_w) < 5:
            logger.debug(f"[MetricSceneGraph3D] Insufficient 3D points for '{label}' ({len(pts_w)} pts). Ignored.")
            return None

        centroid, dimensions, yaw_box = robust_centroid_and_extent(pts_w)

        # Spatial Deduplication & Matching
        best_match_id: Optional[int] = None
        best_dist = float("inf")

        for node_id, node in self.nodes.items():
            if not is_semantic_match(node.label, label):
                continue
            dist = float(np.linalg.norm(node.centroid_world - centroid))
            # Match radius scales with size of the object
            dynamic_radius = max(self.match_radius_base, 0.5 * max(node.dimensions_3d[:2]))
            if dist < dynamic_radius and dist < best_dist:
                best_dist = dist
                best_match_id = node_id

        if best_match_id is not None:
            # Update existing node via Kalman / running average fusion
            node = self.nodes[best_match_id]
            node.observation_count += 1
            node.last_observed_timestamp = timestamp

            # Kalman gain weighting
            alpha = 1.0 / min(node.observation_count, 10)
            node.centroid_world = (1.0 - alpha) * node.centroid_world + alpha * centroid
            node.dimensions_3d = (1.0 - alpha) * node.dimensions_3d + alpha * dimensions
            node.confidence = min(0.99, node.confidence + 0.1 * detection_confidence)

            # Check multi-view confirmation gate (anti-hallucination)
            baseline = float(np.linalg.norm(node.first_viewpoint - uav_position))
            if node.observation_count >= self.min_views_to_confirm and baseline >= self.min_viewpoint_baseline_m:
                node.is_confirmed = True

            return node
        else:
            # Instantiate novel candidate node
            new_node = SceneNode(
                node_id=self.next_node_id,
                label=label,
                centroid_world=centroid,
                dimensions_3d=dimensions,
                confidence=detection_confidence * 0.6,  # Initial candidate discount
                observation_count=1,
                last_observed_timestamp=timestamp,
                first_viewpoint=uav_position.copy(),
                is_confirmed=False,  # Unconfirmed until 2nd viewpoint passes
                yaw_box=yaw_box,
            )
            self.nodes[self.next_node_id] = new_node
            self.next_node_id += 1

            if len(self.nodes) > self.max_nodes:
                self.prune_stale(max_age_s=120.0)

            return new_node

    def get_confirmed_nodes(self) -> List[SceneNode]:
        """Returns all confirmed grounded landmarks."""
        return [node for node in self.nodes.values() if node.is_confirmed]

    def prune_stale(self, max_age_s: float, current_time: Optional[float] = None) -> int:
        """Removes unconfirmed candidate nodes older than max_age_s."""
        if current_time is None:
            if not self.nodes:
                return 0
            current_time = max(n.last_observed_timestamp for n in self.nodes.values())

        to_remove = [
            nid for nid, n in self.nodes.items()
            if not n.is_confirmed and (current_time - n.last_observed_timestamp) > max_age_s
        ]
        for nid in to_remove:
            del self.nodes[nid]
        return len(to_remove)

    def serialize_for_prompt(self, telemetry: DroneTelemetry, max_nodes: int = 15) -> str:
        """Serializes grounded scene graph nodes into structured prompt block."""
        confirmed = self.get_confirmed_nodes()
        if not confirmed:
            return "[3D SCENE GRAPH]\n  No confirmed objects mapped yet."

        # Sort by distance to UAV
        uav_pos = telemetry.position
        scored = []
        for n in confirmed:
            dist = float(np.linalg.norm(n.centroid_world - uav_pos))
            scored.append((dist, n))
        scored.sort(key=lambda x: x[0])

        lines = ["[3D SCENE GRAPH - GROUNDED CONFIRMED OBJECTS]"]
        for dist, n in scored[:max_nodes]:
            dx = n.centroid_world[0] - uav_pos[0]
            dy = n.centroid_world[1] - uav_pos[1]
            dz = n.centroid_world[2] - uav_pos[2]
            bearing_deg = math.degrees(math.atan2(dy, dx) - telemetry.yaw)
            bearing_norm = (bearing_deg + 180.0) % 360.0 - 180.0
            elev_deg = math.degrees(math.atan2(dz, max(1.0, math.hypot(dx, dy))))

            lines.append(
                f"- ID #{n.node_id:02d} [{n.label}]: World(X={n.centroid_world[0]:.1f}, "
                f"Y={n.centroid_world[1]:.1f}, Z={n.centroid_world[2]:.1f}m) | "
                f"Rel: Dist={dist:.1f}m, Bearing={bearing_norm:+.1f}°, Elev={elev_deg:+.1f}° | "
                f"Dim: ({n.dimensions_3d[0]:.1f}m x {n.dimensions_3d[1]:.1f}m x {n.dimensions_3d[2]:.1f}m) | "
                f"Conf={n.confidence:.2f}"
            )
        return "\n".join(lines)
