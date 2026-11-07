"""
Candidate-Proposal Ranking Adapter (SmartWay / LaViRA paradigm).
Generates K candidate waypoint rays/points (e.g., Left, Forward-Left, Center, Forward-Right, Right)
annotated with visual or numeric labels, and prompts the VLM to rank or pick the best candidate.
"""
import json
import re
import math
from typing import Dict, Any, Optional, List, Tuple
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_vlm import BaseVLMBackend
from framework.types import VLNObservation, NavAction
from framework.perception.transforms import (
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame
)


class WaypointCandidateAdapter(BaseVLNAdapter):
    """
    Candidate waypoint proposal and selection adapter.
    Generates K candidate directions in screen-space, probes their clearances via depth,
    and asks the VLM to choose the safest/most aligned candidate ID.
    """
    def __init__(
        self,
        vlm_backend: BaseVLMBackend,
        intrinsics: Optional[CameraIntrinsics] = None,
        num_candidates: int = 5,
        max_range_m: float = 7.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="WaypointCandidateAdapter", config=config)
        self.vlm = vlm_backend
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.num_candidates = num_candidates
        self.max_range_m = max_range_m

    def reset(self, initial_observation: VLNObservation) -> None:
        pass

    def _generate_candidates(self, depth_map) -> List[Dict[str, Any]]:
        """Generates K uniformly distributed candidate bearing rays across the camera view."""
        candidates = []
        u_samples = [0.15, 0.32, 0.50, 0.68, 0.85]
        labels = ["Far-Left", "Mid-Left", "Center-Forward", "Mid-Right", "Far-Right"]

        for idx, (u, label) in enumerate(zip(u_samples, labels)):
            d_m = query_metric_depth_patch(depth_map, u, 0.5, default_depth=self.max_range_m)
            candidates.append({
                "id": idx + 1,
                "label": label,
                "u": u,
                "v": 0.5,
                "depth_m": round(d_m, 2),
                "clearance": "open" if d_m > 4.0 else "obstacle_near"
            })
        return candidates

    def step(self, observation: VLNObservation) -> NavAction:
        candidates = self._generate_candidates(observation.depth_map)

        candidate_text = "\n".join([
            f"  Candidate {c['id']} ({c['label']}): screen_x={c['u']:.2f}, depth={c['depth_m']}m ({c['clearance']})"
            for c in candidates
        ])

        system_prompt = (
            "You are an aerial drone vision navigator.\n"
            "Below is a list of proposed candidate waypoints currently visible from the drone's forward camera.\n"
            "Select the best candidate ID that safely advances toward the target landmark described in the instruction.\n"
            "Output JSON with keys: 'selected_id' (integer 1-5), 'is_stop' (boolean), 'reason' (string)."
        )

        user_content = (
            f"Instruction: \"{observation.instruction}\"\n"
            f"Step: {observation.step_id}\n"
            f"Candidates:\n{candidate_text}\n\n"
            f"Output your choice in JSON format:"
        )

        user_msg: Dict[str, Any] = {"role": "user", "content": user_content}
        if observation.rgb_b64:
            user_msg["images"] = [observation.rgb_b64]

        response = self.vlm.chat(
            messages=[{"role": "system", "content": system_prompt}, user_msg],
            max_tokens=200
        )

        # Parse choice
        choice_id = 3  # default Center-Forward
        is_stop = False
        reason = "Following center corridor"

        m = re.search(r'\{[^{}]*\}', response.content)
        if m:
            try:
                data = json.loads(m.group(0))
                choice_id = int(data.get("selected_id", 3))
                is_stop = bool(data.get("is_stop", False))
                reason = str(data.get("reason", reason))
            except Exception:
                pass

        # Clamp choice to valid index
        choice_id = max(1, min(len(candidates), choice_id))
        chosen = candidates[choice_id - 1]

        # Deproject chosen candidate to 3D body displacement
        dx, dy, dz, yaw_offset = backproject_pixel_to_body_frame(
            u=chosen["u"],
            v=chosen["v"],
            depth_m=chosen["depth_m"],
            intrinsics=self.intrinsics,
            max_range_m=self.max_range_m
        )

        return NavAction(
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_delta=yaw_offset,
            u=chosen["u"],
            v=chosen["v"],
            depth_m=chosen["depth_m"],
            is_stop=is_stop,
            reasoning=f"Selected candidate {choice_id} ({chosen['label']}): {reason}",
            latency_s=response.latency_s,
            metadata={"candidate_id": choice_id, "candidates": candidates}
        )
