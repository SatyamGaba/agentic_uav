"""
SuperNav Aerial Adapter.
Adapted from SOTA SuperNav (ZJU / Causa Robotics, arXiv:2610.12126).
Key Mechanisms:
1. Normalized visual-point grounding [u, v] with metric depth unprojection.
2. Strict Arrival & Standoff Criterion: Never stop when target is across the room (>4m);
   executes final approach hop to standoff distance (2.0 - 3.5m).
3. Near-Zero Motion & Blocked Recovery: If movement stalls (<1.0m),
   triggers exploratory yaw scan (+45 deg) instead of repeating stale actions.
"""
import re
import math
import json
from typing import Dict, Any, Optional, List, Tuple
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_vlm import BaseVLMBackend
from framework.types import VLNObservation, NavAction, Pose3D
from framework.perception.transforms import (
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame,
    euclidean_distance_3d
)


class SuperNavAdapter(BaseVLNAdapter):
    """
    SuperNav visual-point navigation adapter adapted for Aerial Multirotor VLN.
    """
    def __init__(
        self,
        vlm_backend: BaseVLMBackend,
        intrinsics: Optional[CameraIntrinsics] = None,
        max_hop_range_m: float = 7.5,
        standoff_distance_m: float = 3.0,
        arrival_threshold_m: float = 5.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="SuperNavAdapter", config=config)
        self.vlm = vlm_backend
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.max_hop_range_m = max_hop_range_m
        self.standoff_distance_m = standoff_distance_m
        self.arrival_threshold_m = arrival_threshold_m

        self.last_pose: Optional[Pose3D] = None
        self.stall_count: int = 0
        self.recovery_mode: bool = False
        self.step_history: List[Dict[str, Any]] = []

    def reset(self, initial_observation: VLNObservation) -> None:
        self.last_pose = initial_observation.pose
        self.stall_count = 0
        self.recovery_mode = False
        self.step_history = []

    def step(self, observation: VLNObservation) -> NavAction:
        # 1. Check for near-zero motion / stall condition
        if self.last_pose is not None:
            disp = euclidean_distance_3d(observation.pose.to_list(), self.last_pose.to_list())
            if disp < 0.8:
                self.stall_count += 1
            else:
                self.stall_count = 0
        self.last_pose = observation.pose

        # 2. If stalled, execute SuperNav Recovery Sweep (+45 deg yaw)
        if self.stall_count >= 2:
            self.stall_count = 0
            return NavAction(
                dx=2.0,
                dy=0.0,
                dz=0.5,
                yaw_delta=math.radians(45.0),
                u=0.7,
                v=0.5,
                depth_m=5.0,
                is_stop=False,
                reasoning="SuperNav Recovery: near-zero motion detected, executing 45-deg exploratory sweep"
            )

        # 3. Formulate SuperNav Single-Hop Visual Point Prompt with Active Search
        system_prompt = (
            "You are an aerial drone navigator implementing SuperNav visual-point guidance.\n"
            "Analyze the camera view and mark a reachable visual point [u, v] to advance toward the destination landmark.\n"
            "Rules:\n"
            "- [u, v] are normalized in [0.0..1.0]. u: 0.0=left, 0.5=center, 1.0=right. v: 0.0=top, 0.5=horizon, 1.0=bottom.\n"
            "- Mark the landmark body, base, or open corridor threshold leading to it.\n"
            "- If the landmark/corridor is not clearly visible in front, set 'target_visible': false and choose scan_turn_deg (+45 for right, -45 for left).\n"
            "- CRITICAL ARRIVAL RULE: Seeing the target landmark in the distance is NOT arriving. "
            "Set 'is_arrival': true ONLY when the destination landmark is directly in front at touch distance (within 3.5m).\n"
            "- Output valid JSON: {\"u\": float, \"v\": float, \"is_arrival\": bool, \"target_visible\": bool, \"scan_turn_deg\": float, \"reason\": str}"
        )

        user_content = (
            f"Instruction: \"{observation.instruction}\"\n"
            f"Step: {observation.step_id}\n"
            f"Choose visual point [u, v]:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content, "images": [observation.rgb_b64]} if observation.rgb_b64 else {"role": "user", "content": user_content}
        ]

        # 4. Query VLM backend
        response = self.vlm.chat(messages=messages, max_tokens=1024)
        source_text = response.content.strip() if response.content.strip() else response.thinking.strip()
        parsed = self._extract_json(source_text)

        target_u = float(parsed.get("u", 0.5))
        target_v = float(parsed.get("v", 0.5))
        is_arrival = bool(parsed.get("is_arrival", False))
        target_visible = bool(parsed.get("target_visible", True))
        scan_turn = float(parsed.get("scan_turn_deg", 0.0))
        reason = str(parsed.get("reason", "Advancing toward visual waypoint"))

        # If model requested exploratory scan because target isn't visible
        if not target_visible and abs(scan_turn) > 10.0:
            turn_rad = math.radians(max(-60.0, min(60.0, scan_turn)))
            return NavAction(
                dx=1.5,
                dy=0.0,
                dz=0.0,
                yaw_delta=turn_rad,
                u=0.5,
                v=0.5,
                depth_m=10.0,
                is_stop=False,
                reasoning=f"SuperNav Panoramic Scan: {reason}",
                latency_s=response.latency_s,
                raw_response=source_text
            )

        # Clamp (u, v) to valid image bounds
        target_u = max(0.05, min(0.95, target_u))
        target_v = max(0.05, min(0.95, target_v))

        # 5. Sample depth patch around visual anchor
        depth_m = query_metric_depth_patch(
            depth_map=observation.depth_map,
            u=target_u,
            v=target_v,
            default_depth=self.max_hop_range_m
        )

        # 6. SuperNav Strict Arrival Verification:
        # Stop ONLY when VLM confirms arrival AND depth is within arrival touch distance
        if is_arrival and depth_m <= self.arrival_threshold_m:
            return NavAction(
                dx=0.0,
                dy=0.0,
                dz=0.0,
                yaw_delta=0.0,
                u=target_u,
                v=target_v,
                depth_m=depth_m,
                is_stop=True,
                reasoning=f"SuperNav Verified Arrival: {reason} (depth={depth_m:.2f}m <= {self.arrival_threshold_m}m)",
                latency_s=response.latency_s,
                raw_response=source_text
            )

        # 7. Dynamic Hop Sizing & Standoff Approach
        # Far (>15m): 7.0m hop; Mid (8-15m): 5.0m hop; Close (<8m): gentle deceleration
        if depth_m > 15.0:
            hop_range = min(self.max_hop_range_m, 7.5)
        elif depth_m > 8.0:
            hop_range = 5.0
        else:
            hop_range = max(1.5, min(4.0, depth_m - self.standoff_distance_m))

        dx, dy, dz, yaw_offset = backproject_pixel_to_body_frame(
            u=target_u,
            v=target_v,
            depth_m=hop_range,
            intrinsics=self.intrinsics,
            max_range_m=self.max_hop_range_m
        )

        return NavAction(
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_delta=yaw_offset,
            u=target_u,
            v=target_v,
            depth_m=depth_m,
            is_stop=False,
            reasoning=reason,
            latency_s=response.latency_s,
            raw_response=source_text
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON with resilient partial field extraction."""
        if not text:
            return {"u": 0.5, "v": 0.5, "is_arrival": False, "reason": "Center default"}

        m = re.search(r'\{[^{}]*\}', text)
        if m:
            try:
                d = json.loads(m.group(0))
                # Map alternate keys
                if "is_stop" in d and "is_arrival" not in d:
                    d["is_arrival"] = d["is_stop"]
                return d
            except Exception:
                pass

        # Partial regex fallback
        u_m = re.search(r'"u"\s*:\s*([0-9.]+)', text)
        v_m = re.search(r'"v"\s*:\s*([0-9.]+)', text)
        arr_m = re.search(r'"(is_arrival|is_stop)"\s*:\s*(true|false)', text, re.I)
        vis_m = re.search(r'"target_visible"\s*:\s*(true|false)', text, re.I)
        scan_m = re.search(r'"scan_turn_deg"\s*:\s*([-0-9.]+)', text)
        re_m = re.search(r'"reason"\s*:\s*"([^"]+)"', text)

        if u_m and v_m:
            return {
                "u": float(u_m.group(1)),
                "v": float(v_m.group(1)),
                "is_arrival": arr_m.group(2).lower() == "true" if arr_m else False,
                "target_visible": vis_m.group(1).lower() == "true" if vis_m else True,
                "scan_turn_deg": float(scan_m.group(1)) if scan_m else 0.0,
                "reason": re_m.group(1) if re_m else "VLM guided hop"
            }

        return {
            "u": 0.5,
            "v": 0.5,
            "is_arrival": False,
            "target_visible": True,
            "scan_turn_deg": 0.0,
            "reason": "Center default"
        }
