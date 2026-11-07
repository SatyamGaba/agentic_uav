import cv2
import time
import math
import json
import logging
import numpy as np
import zmq
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger("MavicVLM")

class PySLAMClient:
    """High-speed ZeroMQ Client bridging Windows Webots controller to Linux Docker pySLAM server.
    
    Streams JPEG encoded camera frames over TCP port 5555 and receives 6-DoF pose matrices (T_w_c)
    and 3D map point clouds in real time.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_ms: int = 500):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket = None
        self.connected = False
        
        self.current_pose = np.eye(4, dtype=np.float64)
        self.map_points: List[np.ndarray] = []
        self.last_sync_time = 0.0
        
        # Camera Intrinsics matching Webots Camera (640x480, FOV=0.785398 rad)
        self.fx = 772.0
        self.fy = 772.0
        self.cx = 320.0
        self.cy = 240.0
        
        self._connect()

    def _connect(self) -> None:
        try:
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.connected = True
            logger.info(f"[PySLAM Client] Connected to Docker SLAM Server at tcp://{self.host}:{self.port}")
        except Exception as e:
            logger.warning(f"[PySLAM Client] Connection failed to tcp://{self.host}:{self.port}: {e}")
            self.connected = False

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: float,
        telemetry: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Encodes frame as JPEG and streams to Docker pySLAM server over ZMQ TCP.
        
        Returns:
            pose: 4x4 SE(3) transformation matrix (T_w_c)
            map_points: List of 3D point arrays [x, y, z] in world coordinates
        """
        if not self.connected or self.socket is None or frame_bgr is None:
            return self.current_pose, self.map_points

        try:
            # Compress image to high-quality JPEG (95%) to preserve ORB features while optimizing network speed
            success, encoded_img = cv2.imencode('.jpg', frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not success:
                return self.current_pose, self.map_points

            meta = {"timestamp": timestamp}
            if telemetry:
                meta.update(telemetry)
            meta_bytes = json.dumps(meta).encode('utf-8')
            
            # Send multipart message: [jpeg_bytes, meta_bytes]
            self.socket.send_multipart([encoded_img.tobytes(), meta_bytes])
            
            # Receive response JSON
            resp = self.socket.recv_json()
            
            if "pose" in resp:
                self.current_pose = np.array(resp["pose"], dtype=np.float64)
            
            if "map_points" in resp:
                pts_list = resp["map_points"]
                self.map_points = [np.array(pt, dtype=np.float64) for pt in pts_list]
                
            self.last_sync_time = timestamp
            logger.debug(f"[PySLAM Client] Synced step {resp.get('step', 0)}: {len(self.map_points)} 3D points tracked.")

        except zmq.Again:
            logger.debug("[PySLAM Client] ZMQ timeout waiting for Docker pySLAM response.")
            self._reconnect()
        except Exception as e:
            logger.warning(f"[PySLAM Client] Frame processing sync error: {e}")
            self._reconnect()

        return self.current_pose, self.map_points

    def _reconnect(self) -> None:
        """Re-establishes ZMQ REQ socket connection to clear invalid state."""
        try:
            if self.socket:
                self.socket.close(linger=0)
            self.socket = self.context.socket(zmq.REQ)
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.connect(f"tcp://{self.host}:{self.port}")
            self.connected = True
        except Exception as e:
            logger.warning(f"[PySLAM Client] Reconnect failed: {e}")
            self.connected = False

    def get_camera_frame_points(self) -> List[np.ndarray]:
        """Returns the current map points transformed into the camera's local coordinate frame."""
        if not self.map_points:
            return []
            
        # Compute T_c_w from current_pose T_w_c
        R_w_c = self.current_pose[:3, :3]
        t_w_c = self.current_pose[:3, 3]
        R_c_w = R_w_c.T
        t_c_w = -R_c_w @ t_w_c
        
        cam_pts = []
        for pt3d_w in self.map_points:
            pt3d_c = R_c_w @ pt3d_w + t_c_w
            cam_pts.append(pt3d_c)
            
        return cam_pts


    def get_map_points_in_bbox(
        self,
        pixel_bbox: Tuple[float, float, float, float]
    ) -> List[np.ndarray]:
        """Finds 3D map points that project into a 2D bounding box [u_min, v_min, u_max, v_max]."""
        if not self.map_points:
            return []

        u_min, v_min, u_max, v_max = pixel_bbox
        K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)

        # Compute T_c_w from current_pose T_w_c
        R_w_c = self.current_pose[:3, :3]
        t_w_c = self.current_pose[:3, 3]
        R_c_w = R_w_c.T
        t_c_w = -R_c_w @ t_w_c

        matched_pts: List[np.ndarray] = []
        for pt3d_w in self.map_points:
            pt3d_c = R_c_w @ pt3d_w + t_c_w
            if pt3d_c[2] <= 0.1:
                continue
                
            uv = K @ pt3d_c
            if uv[2] == 0:
                continue
            u, v = uv[0] / uv[2], uv[1] / uv[2]
            
            if u_min <= u <= u_max and v_min <= v <= v_max:
                matched_pts.append(pt3d_w)

        return matched_pts

    def close(self) -> None:
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
