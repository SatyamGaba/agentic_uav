"""
High-Performance Closed-Loop Aerial VLN Adapter (SOTA Architecture).
Maximizes OSR & SR, Minimizes Navigation Error (NE) through:
1. Sub-Goal Landmark Chunking: Prevents attention drift on long composite instructions.
2. Step-by-Step Closed-Loop Visual Servoing: Re-observes fresh RGB camera frames at EVERY step.
3. Continuous 2D-to-3D Geometric Grounding: Deprojects predicted landmark coordinates (u, v) into continuous 3D body displacements (dx, dy, dz, yaw_delta), eliminating 30-degree discretization deficit.
4. Multi-Patch Depth Probing & Arrival Gating: 9-point spatial depth inspection across camera FOV to detect off-center landmarks and trigger precision stopping (<3.5m).
"""
import re
import math
import json
from typing import Dict, Any, Optional, List, Tuple
import numpy as np

from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_vlm import BaseVLMBackend
from framework.types import VLNObservation, NavAction, Pose3D
from framework.perception.transforms import (
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame
)
from framework.adapters.discrete_action import OPENFLY_ACTION_VOCAB


CLOSED_LOOP_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "navigate_step",
            "description": "Visual servoing step toward the active navigation landmark or flight corridor.",
            "parameters": {
                "type": "object",
                "properties": {
                    "u": {
                        "type": "number",
                        "description": "Horizontal pixel coordinate of landmark/path in camera FOV [0.0=far left, 0.5=center, 1.0=far right]"
                    },
                    "v": {
                        "type": "number",
                        "description": "Vertical pixel coordinate [0.0=climb/top, 0.5=horizon/level, 1.0=descend/ground]"
                    },
                    "step_distance_m": {
                        "type": "number",
                        "description": "Forward step displacement in meters [2.0 to 9.0m]"
                    },
                    "is_stop": {
                        "type": "boolean",
                        "description": "Set to true ONLY if the current landmark or final destination has been reached"
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation of visible landmark and chosen flight trajectory"
                    }
                },
                "required": ["u", "v", "step_distance_m", "is_stop", "reasoning"]
            }
        }
    }
]


class CalibratedNavAdapter(BaseVLNAdapter):
    """
    End-to-End Closed-Loop Aerial VLN Adapter.
    Performs visual perception, landmark grounding, and continuous steering at every step.
    Supports both smooth continuous kinematics and canonical OpenFly discrete action spaces.
    """
    def __init__(
        self,
        vlm_backend: BaseVLMBackend,
        intrinsics: Optional[CameraIntrinsics] = None,
        mode: str = "continuous",  # 'continuous' for exact 3D visual servoing, 'discrete' for OpenFly action codes
        max_range_m: float = 9.0,
        stop_proximity_m: float = 3.5,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="CalibratedNavAdapter", config=config)
        if vlm_backend is None:
            raise ValueError("CalibratedNavAdapter strictly requires an active VLM backend. Heuristic rule fallbacks have been permanently removed.")
        self.vlm = vlm_backend
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.mode = mode
        self.max_range_m = max_range_m
        self.stop_proximity_m = stop_proximity_m

        self.subgoals: List[str] = []
        self.current_subgoal_idx: int = 0
        self.steps_in_subgoal: int = 0
        self.step_history: List[Dict[str, Any]] = []

    def reset(self, initial_observation: VLNObservation) -> None:
        self.subgoals = self._chunk_instruction(initial_observation.instruction)
        self.current_subgoal_idx = 0
        self.steps_in_subgoal = 0
        self.step_history.clear()

    def _probe_spatial_depth(self, depth_map: np.ndarray) -> Dict[str, float]:
        """Probes a 9-point spatial depth grid across the camera FOV to detect obstacles and off-center landmarks."""
        points = {
            "center": (0.5, 0.5),
            "left": (0.25, 0.5),
            "right": (0.75, 0.5),
            "top": (0.5, 0.3),
            "bottom": (0.5, 0.7),
            "top_left": (0.25, 0.3),
            "top_right": (0.75, 0.3),
            "bottom_left": (0.25, 0.7),
            "bottom_right": (0.75, 0.7)
        }
        res = {}
        for name, (u, v) in points.items():
            res[name] = query_metric_depth_patch(depth_map, u, v, default_depth=self.max_range_m)
        res["min_frontal"] = min(res["center"], res["left"], res["right"], res["top"], res["bottom"])
        return res

    def step(self, observation: VLNObservation) -> NavAction:
        self.steps_in_subgoal += 1
        is_final_subgoal = (self.current_subgoal_idx >= len(self.subgoals) - 1)

        # 1. Multi-Patch Spatial Depth Probing
        depth_info = self._probe_spatial_depth(observation.depth_map)

        # 2. Dual-Gated Metric Stop Check on Final Approach
        if is_final_subgoal and depth_info["min_frontal"] <= self.stop_proximity_m:
            return NavAction(
                dx=0.0, dy=0.0, dz=0.0, yaw_delta=0.0,
                u=0.5, v=0.5, depth_m=depth_info["min_frontal"],
                action_code=0,
                is_stop=True,
                reasoning=f"Multi-Patch Stop: Final landmark reached (frontal depth={depth_info['min_frontal']:.1f}m <= {self.stop_proximity_m}m)",
                metadata={"gated_arrival": True, "depth_grid": depth_info}
            )

        # 3. Active Landmark Instruction
        if self.subgoals and self.current_subgoal_idx < len(self.subgoals):
            active_subgoal = self.subgoals[self.current_subgoal_idx]
        else:
            active_subgoal = observation.instruction

        # 4. Live Visual Perception & Reasoning Call
        target_u, target_v, step_m, vlm_stop, reason = self._visual_grounding_step(
            observation=observation,
            subgoal=active_subgoal,
            is_final=is_final_subgoal,
            depth_info=depth_info
        )

        # 5. Handle Landmark Subgoal Progression
        if vlm_stop:
            if not is_final_subgoal:
                # Milestone reached: advance to next landmark in instruction
                self.current_subgoal_idx += 1
                self.steps_in_subgoal = 0
                # Re-query heading for new landmark or step forward into corridor
                target_u, target_v, step_m, _, reason = self._visual_grounding_step(
                    observation=observation,
                    subgoal=self.subgoals[self.current_subgoal_idx],
                    is_final=(self.current_subgoal_idx >= len(self.subgoals) - 1),
                    depth_info=depth_info
                )
            else:
                # Final destination reached
                return NavAction(
                    dx=0.0, dy=0.0, dz=0.0, yaw_delta=0.0,
                    u=target_u, v=target_v,
                    depth_m=depth_info["center"],
                    action_code=0,
                    is_stop=True,
                    reasoning=f"Visual Arrival: {reason}",
                    metadata={"subgoal_idx": self.current_subgoal_idx}
                )

        # 6. Metric Depth Deprojection & Collision Shielding
        depth_at_target = query_metric_depth_patch(
            depth_map=observation.depth_map,
            u=target_u,
            v=target_v,
            default_depth=self.max_range_m
        )

        # Deproject (u, v) into continuous 3D body displacements
        dx, dy, dz, yaw_offset = backproject_pixel_to_body_frame(
            u=target_u,
            v=target_v,
            depth_m=depth_at_target,
            intrinsics=self.intrinsics,
            max_range_m=min(self.max_range_m, max(2.0, step_m))
        )

        # Terminal deceleration check
        if is_final_subgoal and depth_info["center"] < 8.0:
            dx = min(dx, max(1.5, depth_info["center"] - 1.5))

        # 7. Action Space Formatting (Continuous vs Calibrated Discrete)
        action_code = None
        if self.mode == "discrete":
            action_code = self._map_to_discrete_action(dx, dy, dz, yaw_offset)

        action = NavAction(
            dx=dx,
            dy=dy,
            dz=dz,
            yaw_delta=yaw_offset,
            u=target_u,
            v=target_v,
            depth_m=depth_at_target,
            action_code=action_code,
            is_stop=False,
            reasoning=f"Closed-loop step toward [{active_subgoal[:35]}]: {reason}",
            metadata={
                "subgoal_idx": self.current_subgoal_idx,
                "subgoal_total": len(self.subgoals),
                "target_u": target_u,
                "target_v": target_v,
                "mode": self.mode
            }
        )

        self.step_history.append({"subgoal": active_subgoal, "u": target_u, "v": target_v, "dx": dx, "dy": dy})
        return action

    def _visual_grounding_step(
        self,
        observation: VLNObservation,
        subgoal: str,
        is_final: bool,
        depth_info: Optional[Dict[str, float]] = None
    ) -> Tuple[float, float, float, bool, str]:
        """Executes a single-step live VLM call with visual image and tool calling."""
        stop_prompt = " Set is_stop=true ONLY if this final destination is reached." if is_final else " Set is_stop=true if reached."
        depth_str = ""
        if depth_info:
            depth_str = f"\nDepth Sensor: Center={depth_info.get('center', 10.0):.1f}m, Left={depth_info.get('left', 10.0):.1f}m, Right={depth_info.get('right', 10.0):.1f}m."
        history_str = ""
        if self.step_history:
            last = self.step_history[-1]
            history_str = f"\nPrevious step: moved {last.get('dx', 6.0):.1f}m toward (u={last.get('u', 0.5):.2f}, v={last.get('v', 0.5):.2f})."
        if self.steps_in_subgoal >= 5:
            history_str += f"\nNote: You have taken {self.steps_in_subgoal} steps for this landmark. If reached or passed, set is_stop=true to advance."

        prompt = (
            f"You are an autonomous drone navigator. Active Subgoal: \"{subgoal}\".\n"
            f"Step: {observation.step_id}. Look at the forward camera image.{depth_str}{history_str}\n"
            f"Call navigate_step with target coordinates (u, v) and step distance.{stop_prompt}"
        )

        msg: Dict[str, Any] = {"role": "user", "content": prompt}
        if observation.rgb_b64:
            msg["images"] = [observation.rgb_b64]

        resp = self.vlm.chat(messages=[msg], tools=CLOSED_LOOP_TOOLS, max_tokens=1500)

        # 1. Check native tool calls
        for tc in resp.tool_calls:
            if tc.get("function", {}).get("name") == "navigate_step":
                args = tc["function"].get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                u_val = float(args.get("u", 0.5))
                v_val = float(args.get("v", 0.5))
                step_m = float(args.get("step_distance_m", 6.0))
                is_stop = bool(args.get("is_stop", False))
                reason = str(args.get("reasoning", "Navigating toward landmark"))
                return u_val, v_val, step_m, is_stop, reason

        # 2. Resilient JSON extraction from content or thinking trace
        candidate_text = resp.content.strip() if resp.content.strip() else resp.thinking.strip()
        parsed = self._extract_step_json(candidate_text)
        return (
            parsed.get("u", 0.5),
            parsed.get("v", 0.5),
            parsed.get("step_distance_m", 6.0),
            parsed.get("is_stop", False),
            parsed.get("reasoning", "Visual grounding step")
        )

    def _extract_step_json(self, text: str) -> Dict[str, Any]:
        """Extracts JSON parameters from text or thinking trace."""
        if not text:
            return {"u": 0.5, "v": 0.5, "step_distance_m": 6.0, "is_stop": False, "reasoning": "Center forward"}

        # Try finding standard JSON object
        m = re.search(r'\{[^{}]*\}', text)
        if m:
            try:
                data = json.loads(m.group(0))
                return {
                    "u": float(data.get("u", 0.5)),
                    "v": float(data.get("v", 0.5)),
                    "step_distance_m": float(data.get("step_distance_m", data.get("step_m", 6.0))),
                    "is_stop": bool(data.get("is_stop", False)),
                    "reasoning": str(data.get("reasoning", "Visual grounding step"))
                }
            except Exception:
                pass

        # Regex key-value extraction
        u_m = re.search(r'"?u"?\s*:\s*([0-9.]+)', text)
        v_m = re.search(r'"?v"?\s*:\s*([0-9.]+)', text)
        step_m = re.search(r'"?(?:step_distance_m|step_m)"?\s*:\s*([0-9.]+)', text)
        stop_m = re.search(r'"?is_stop"?\s*:\s*(true|false)', text, re.I)
        reason_m = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', text)

        return {
            "u": float(u_m.group(1)) if u_m else 0.5,
            "v": float(v_m.group(1)) if v_m else 0.5,
            "step_distance_m": float(step_m.group(1)) if step_m else 6.0,
            "is_stop": stop_m.group(1).lower() == "true" if stop_m else False,
            "reasoning": reason_m.group(1) if reason_m else "Visual step"
        }

    def _map_to_discrete_action(self, dx: float, dy: float, dz: float, yaw_delta: float) -> int:
        """Maps continuous 3D body vector to the optimal canonical OpenFly discrete action code."""
        # Significant yaw change takes precedence
        if yaw_delta > math.radians(14.0):
            return 2  # TURN_RIGHT (+30 deg)
        elif yaw_delta < -math.radians(14.0):
            return 3  # TURN_LEFT (-30 deg)

        # Significant altitude change
        if dz > 1.8:
            return 4  # CLIMB (+3m)
        elif dz < -1.8:
            return 5  # DESCEND (-3m)

        # Forward strides
        if dx > 7.5:
            return 9  # MOVE_FORWARD_LARGE (9m)
        elif dx > 4.5:
            return 8  # MOVE_FORWARD_MED (6m)
        else:
            return 1  # MOVE_FORWARD_FINE (3m)

    def _chunk_instruction(self, instruction: str) -> List[str]:
        """Decomposes raw instruction into sequential landmark subgoals without fragmenting compound adjectives."""
        tokens = re.split(r'[;.\n]|\b(?:then|next|finally|after that)\b|\band\s+(?=(?:fly|turn|head|proceed|advance|move|climb|descend|go)\b)', instruction, flags=re.IGNORECASE)
        subgoals = [t.strip() for t in tokens if len(t.strip()) > 5]
        return subgoals if subgoals else [instruction]
