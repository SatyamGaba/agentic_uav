import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger("MavicVLM")

class PySLAMInterface:
    """
    Wrapper for luigifreda/pyslam (https://github.com/luigifreda/pyslam)
    Initializes pySLAM engine, feeds frames, and extracts 6-DoF pose and 3D map points.
    """
    def __init__(self, fx: float, fy: float, cx: float, cy: float, width: int, height: int, feature_tracker: str = 'orb'):
        self.camera_intrinsics = {
            'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy, 'width': width, 'height': height
        }
        self.initialized = False
        self.current_pose = np.eye(4)
        self.map_points = []
        
        try:
            # Attempt to import pyslam components
            # NOTE: User must install luigifreda/pyslam manually
            from pyslam.slam import SLAM
            from pyslam.camera import PinholeCamera
            
            # Initialize camera model
            self.camera = PinholeCamera(width, height, fx, fy, cx, cy, [0, 0, 0, 0])
            
            # Initialize SLAM engine
            self.slam = SLAM(self.camera, feature_tracker=feature_tracker)
            self.initialized = True
            logger.info(f"[PySLAM] Initialized successfully with {feature_tracker.upper()} tracker.")
        except ImportError:
            logger.error("[PySLAM] 'pyslam' module not found. Please install luigifreda/pyslam.")
            self.slam = None

    def process_frame(self, frame_bgr: np.ndarray, timestamp: float) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Feed continuous video frames into pySLAM tracking thread.
        Returns:
            pose: 4x4 SE(3) transformation matrix (T_w_c)
            map_points: List of 3D points [x, y, z] in world coordinates
        """
        if not self.initialized or self.slam is None:
            return self.current_pose, self.map_points

        try:
            # Feed frame to SLAM tracker
            self.slam.track(frame_bgr, timestamp)
            
            # Extract 6-DoF pose (T_w_c)
            if self.slam.pose is not None:
                self.current_pose = self.slam.pose
            
            # Extract triangulated 3D map points from active keyframes
            if hasattr(self.slam, 'map') and self.slam.map is not None:
                active_points = self.slam.map.get_active_points()
                self.map_points = [p.pt3d for p in active_points if p is not None and hasattr(p, 'pt3d') and p.pt3d is not None]
                
            logger.debug(f"[PySLAM] Tracked frame at {timestamp}. Map points: {len(self.map_points)}")
            
        except Exception as e:
            logger.error(f"[PySLAM] Tracking error: {e}")

        return self.current_pose, self.map_points

    def get_map_points_in_bbox(self, pixel_bbox: Tuple[float, float, float, float]) -> List[np.ndarray]:
        """
        Find 3D map points that project into a given 2D bounding box.
        pixel_bbox: [u_min, v_min, u_max, v_max]
        """
        if not self.map_points or not self.initialized:
            return []

        u_min, v_min, u_max, v_max = pixel_bbox
        K = np.array([
            [self.camera_intrinsics['fx'], 0, self.camera_intrinsics['cx']],
            [0, self.camera_intrinsics['fy'], self.camera_intrinsics['cy']],
            [0, 0, 1]
        ])

        matched_pts = []
        # T_w_c -> T_c_w
        R_w_c = self.current_pose[:3, :3]
        t_w_c = self.current_pose[:3, 3]
        R_c_w = R_w_c.T
        t_c_w = -R_c_w @ t_w_c

        for pt3d_w in self.map_points:
            # Transform world point to camera frame
            pt3d_c = R_c_w @ pt3d_w + t_c_w
            
            # Check depth (cheirality)
            if pt3d_c[2] <= 0.1:
                continue
                
            # Project to image plane
            uv = K @ pt3d_c
            if uv[2] == 0:
                continue
            u, v = uv[0]/uv[2], uv[1]/uv[2]
            
            # Check if inside bounding box
            if u_min <= u <= u_max and v_min <= v <= v_max:
                matched_pts.append(pt3d_w)

        return matched_pts
