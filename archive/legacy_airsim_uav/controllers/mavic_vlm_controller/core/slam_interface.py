"""
Known-Pose Sparse Triangulation Mapper
======================================
Tracks 2D ORB visual features across successive keyframes, triangulates 3D points
using UAV telemetry poses, and anchors metric scale without overclaiming full SLAM.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, List, Tuple, Dict, Any
import cv2
import numpy as np

logger = logging.getLogger("MavicVLM")


class KnownPoseSparseMapper:
    """Multi-view known-pose feature triangulator maintaining an 8,000-point world cloud."""

    def __init__(self, max_points: int = 8000, max_features: int = 500, focal_length_px: float = 350.0):
        self.max_points = max_points
        self.max_features = max_features
        self.focal_length = focal_length_px
        self.orb = cv2.ORB_create(nfeatures=max_features)

        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.prev_pose: Optional[np.ndarray] = None  # 4x4 SE(3) T_wc
        self.map_points: List[np.ndarray] = []
        self.keyframes: List[Dict[str, Any]] = []

    def process_frame(
        self,
        gray_image: np.ndarray,
        T_wc: np.ndarray,
        timestamp: float,
    ) -> List[np.ndarray]:
        """Processes frame, tracks features from previous keyframe, and triangulates 3D points."""
        h, w = gray_image.shape[:2]
        cx, cy = w / 2.0, h / 2.0
        K = np.array([
            [self.focal_length, 0.0, cx],
            [0.0, self.focal_length, cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)

        if self.prev_gray is None:
            # First keyframe
            pts = cv2.goodFeaturesToTrack(gray_image, maxCorners=self.max_features, qualityLevel=0.01, minDistance=7)
            self.prev_gray = gray_image
            self.prev_pts = pts
            self.prev_pose = T_wc.copy()
            return []

        # Check baseline displacement
        baseline = float(np.linalg.norm(T_wc[:3, 3] - self.prev_pose[:3, 3]))
        if baseline < 0.8:
            return []

        # Optical flow tracking between keyframes
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray_image, self.prev_pts, None)
        if p1 is not None and len(p1) > 0:
            valid_prev = self.prev_pts[st.flatten() == 1]
            valid_curr = p1[st.flatten() == 1]

            if len(valid_prev) >= 8:
                # Relative camera projection matrices
                T_cw_prev = np.linalg.inv(self.prev_pose)
                T_cw_curr = np.linalg.inv(T_wc)

                P1 = K @ T_cw_prev[:3, :]
                P2 = K @ T_cw_curr[:3, :]

                # Triangulate homogeneous 4D coordinates
                pts4D = cv2.triangulatePoints(P1, P2, valid_prev.T, valid_curr.T)
                pts3D = (pts4D[:3, :] / np.maximum(1e-5, pts4D[3, :])).T

                # Filter valid points in front of cameras
                new_points = []
                for pt in pts3D:
                    dist = float(np.linalg.norm(pt - T_wc[:3, 3]))
                    if 1.0 < dist < 80.0 and not np.isnan(pt).any():
                        new_points.append(pt)
                        self.map_points.append(pt)

                if len(self.map_points) > self.max_points:
                    self.map_points = self.map_points[-self.max_points:]

                # Store keyframe
                self.keyframes.append({
                    "pose": T_wc.copy(),
                    "pts_2d": valid_curr,
                    "pts_3d": np.array(new_points) if new_points else np.empty((0, 3)),
                    "timestamp": timestamp,
                })
                if len(self.keyframes) > 20:
                    self.keyframes = self.keyframes[-20:]

        # Update previous keyframe
        new_features = cv2.goodFeaturesToTrack(gray_image, maxCorners=self.max_features, qualityLevel=0.01, minDistance=7)
        self.prev_gray = gray_image
        self.prev_pts = new_features
        self.prev_pose = T_wc.copy()

        return self.map_points

    def query_bbox_depth(
        self,
        bbox_norm: Tuple[float, float, float, float],
        img_w: int = 400,
        img_h: int = 300,
    ) -> Optional[float]:
        """Queries median depth from map points projecting within normalized 2D bounding box."""
        if not self.keyframes or not self.map_points:
            return None

        u_min = bbox_norm[0] * img_w
        v_min = bbox_norm[1] * img_h
        u_max = bbox_norm[2] * img_w
        v_max = bbox_norm[3] * img_h

        matched_depths: List[float] = []
        for kf in reversed(self.keyframes[-5:]):
            pts_2d = kf["pts_2d"]
            pts_3d = kf["pts_3d"]
            cam_pos = kf["pose"][:3, 3]

            for i in range(len(pts_2d)):
                u, v = pts_2d[i][0], pts_2d[i][1]
                if u_min <= u <= u_max and v_min <= v <= v_max:
                    dist = float(np.linalg.norm(pts_3d[i] - cam_pos))
                    matched_depths.append(dist)

        if len(matched_depths) >= 3:
            return float(np.median(matched_depths))
        return None

    def point_cloud(self) -> np.ndarray:
        if not self.map_points:
            return np.empty((0, 3), dtype=np.float64)
        return np.array(self.map_points, dtype=np.float64)
