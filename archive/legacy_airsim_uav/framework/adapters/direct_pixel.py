"""
Direct Continuous Pixel Grounding Adapter (See-Point-Fly paradigm).
Instructs the VLM to predict continuous visual waypoint coordinates (u, v) on the camera frame,
then deprojects to metric 3D flight displacements using camera geometry and on-demand depth.
"""
import json
import re
import math
from typing import Dict, Any, Optional
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_vlm import BaseVLMBackend
from framework.types import VLNObservation, NavAction
from framework.perception.transforms import (
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame
)


class DirectPixelAdapter(BaseVLNAdapter):
    """
    Continuous visual grounding adapter.
    Single-turn VLM prompt outputting normalized (u, v) visual target.
    Eliminates multi-turn tool-calling schema overhead and avoids thinking token traps.
    """
    def __init__(
        self,
        vlm_backend: BaseVLMBackend,
        intrinsics: Optional[CameraIntrinsics] = None,
        max_range_m: float = 8.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="DirectPixelAdapter", config=config)
        self.vlm = vlm_backend
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.max_range_m = max_range_m

    def reset(self, initial_observation: VLNObservation) -> None:
        pass

    def step(self, observation: VLNObservation) -> NavAction:
        system_prompt = (
            "You are an aerial drone vision navigator.\n"
            "Analyze the onboard camera view and choose a safe waypoint to proceed toward the destination landmark.\n"
            "Output ONLY valid JSON with keys: 'u', 'v', 'is_stop', 'reason'.\n"
            "- u: horizontal coordinate [0.0=left, 0.5=center, 1.0=right]\n"
            "- v: vertical coordinate [0.0=top, 0.5=horizon, 1.0=bottom]\n"
            "- is_stop: true ONLY if the final destination is visually reached and in front\n"
            "- reason: concise 1-sentence description"
        )

        user_content = (
            f"Instruction: \"{observation.instruction}\"\n"
            f"Step: {observation.step_id}\n"
            f"Output the visual waypoint JSON:"
        )

        user_msg: Dict[str, Any] = {"role": "user", "content": user_content}
        if observation.rgb_b64:
            user_msg["images"] = [observation.rgb_b64]

        messages = [
            {"role": "system", "content": system_prompt},
            user_msg
        ]

        # Request up to 1536 tokens to accommodate VLM reasoning/thinking trace
        response = self.vlm.chat(messages=messages, max_tokens=1536)
        
        # Check content first, fall back to thinking trace if content is empty
        source_text = response.content.strip() if response.content.strip() else response.thinking.strip()
        parsed = self._extract_json(source_text)

        target_u = float(parsed.get("u", 0.5))
        target_v = float(parsed.get("v", 0.5))
        is_stop = bool(parsed.get("is_stop", False))
        reason = str(parsed.get("reason", "Proceeding toward visual waypoint"))

        # Query metric depth at selected pixel
        depth_m = query_metric_depth_patch(
            depth_map=observation.depth_map,
            u=target_u,
            v=target_v,
            default_depth=self.max_range_m
        )

        # Proximity arrival check if stop was signaled
        if is_stop and depth_m < 4.0:
            return NavAction(
                u=target_u,
                v=target_v,
                depth_m=depth_m,
                is_stop=True,
                reasoning=reason,
                latency_s=response.latency_s
            )

        # Back-project into 3D body displacement
        dx, dy, dz, yaw_offset = backproject_pixel_to_body_frame(
            u=target_u,
            v=target_v,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            max_range_m=self.max_range_m
        )

        return NavAction(
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_delta=yaw_offset,
            u=target_u,
            v=target_v,
            depth_m=depth_m,
            is_stop=is_stop,
            reasoning=reason,
            latency_s=response.latency_s,
            raw_response=source_text
        )

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON object or key-values from response/thinking string."""
        if not text:
            return {"u": 0.5, "v": 0.5, "is_stop": False, "reason": "Default center path"}

        # 1. Try finding complete JSON block
        m = re.search(r'\{[^{}]*\}', text)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass

        # 2. Resilient partial extractor: extract individual fields if token budget was cut off
        u_match = re.search(r'"u"\s*:\s*([0-9.]+)', text)
        v_match = re.search(r'"v"\s*:\s*([0-9.]+)', text)
        stop_match = re.search(r'"is_stop"\s*:\s*(true|false)', text, re.IGNORECASE)
        reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', text)

        if u_match and v_match:
            return {
                "u": float(u_match.group(1)),
                "v": float(v_match.group(1)),
                "is_stop": stop_match.group(1).lower() == "true" if stop_match else False,
                "reason": reason_match.group(1) if reason_match else "VLM guided step"
            }

        return {"u": 0.5, "v": 0.5, "is_stop": False, "reason": "Default center path"}
