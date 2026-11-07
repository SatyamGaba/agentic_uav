"""
AgenticNav Tool-Calling Adapter.
Formulates Aerial VLN as an interactive tool-calling harness:
- move_to(u, v, reason): direct pixel-level grounding
- query_depth(points): on-demand localized metric depth inspection
- recall_memory(step_id): agentic history retrieval
- stop_arrival(u, v, reason): verified destination termination
"""
import json
import math
from typing import Dict, Any, Optional, List, Tuple
from framework.interfaces.base_adapter import BaseVLNAdapter
from framework.interfaces.base_vlm import BaseVLMBackend
from framework.types import VLNObservation, NavAction, Pose3D
from framework.perception.transforms import (
    CameraIntrinsics,
    query_metric_depth_patch,
    backproject_pixel_to_body_frame
)


AGENTIC_NAV_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_to",
            "description": "Select a normalized coordinate (u, v) in camera frame [0.0..1.0] to navigate forward.",
            "parameters": {
                "type": "object",
                "properties": {
                    "u": {"type": "number", "description": "Horizontal coordinate [0.0=left, 0.5=center, 1.0=right]"},
                    "v": {"type": "number", "description": "Vertical coordinate [0.0=top, 0.5=horizon, 1.0=bottom]"},
                    "reason": {"type": "string", "description": "Visual explanation of the chosen corridor or landmark"}
                },
                "required": ["u", "v", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_depth",
            "description": "Query precise metric depth (meters) for a batch of coordinates before deciding.",
            "parameters": {
                "type": "object",
                "properties": {
                    "points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "u": {"type": "number", "description": "Horizontal fraction [0..1]"},
                                "v": {"type": "number", "description": "Vertical fraction [0..1]"}
                            },
                            "required": ["u", "v"]
                        },
                        "description": "List of (u, v) points to probe"
                    }
                },
                "required": ["points"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_arrival",
            "description": "Signal arrival when the final destination landmark is clearly visible directly in front.",
            "parameters": {
                "type": "object",
                "properties": {
                    "u": {"type": "number", "description": "Horizontal landmark coordinate [0..1]"},
                    "v": {"type": "number", "description": "Vertical landmark coordinate [0..1]"},
                    "reason": {"type": "string", "description": "Visual identification of the final destination"}
                },
                "required": ["u", "v", "reason"]
            }
        }
    }
]


class AgenticNavAdapter(BaseVLNAdapter):
    """
    Full AgenticNav interactive tool-calling harness.
    """
    def __init__(
        self,
        vlm_backend: BaseVLMBackend,
        intrinsics: Optional[CameraIntrinsics] = None,
        max_substeps: int = 3,
        max_range_m: float = 8.0,
        config: Optional[Dict[str, Any]] = None
    ):
        super().__init__(name="AgenticNavAdapter", config=config)
        self.vlm = vlm_backend
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.max_substeps = max_substeps
        self.max_range_m = max_range_m
        self.memory_history: List[Dict[str, Any]] = []

    def reset(self, initial_observation: VLNObservation) -> None:
        self.memory_history.clear()

    def step(self, observation: VLNObservation) -> NavAction:
        system_prompt = (
            "You are AgenticNav, an autonomous 3D aerial drone navigator.\n"
            "You see the live forward onboard camera view. Follow the instruction to reach the destination landmark.\n\n"
            "Action Rules:\n"
            "1. Call move_to(u, v, reason) to fly toward an open visual path or visible landmark (u in [0..1], v in [0..1]).\n"
            "2. Call query_depth(points) to probe distances before committing if uncertain.\n"
            "3. Call stop_arrival(u, v, reason) ONLY when the final landmark is reached and in front.\n"
            "Keep thinking brief. Immediately emit your tool call."
        )

        user_content = (
            f"Navigation Instruction: \"{observation.instruction}\"\n"
            f"Step: {observation.step_id} | Pose: X={observation.pose.x:.1f}, Y={observation.pose.y:.1f}, Z={observation.pose.z:.1f} | Yaw={math.degrees(observation.pose.yaw):.1f} deg\n"
            f"Inspect the onboard camera frame and output your action tool call now."
        )

        user_msg: Dict[str, Any] = {"role": "user", "content": user_content}
        if observation.rgb_b64:
            user_msg["images"] = [observation.rgb_b64]

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            user_msg
        ]

        recorded_tool_calls = []
        action_out: Optional[NavAction] = None

        for substep in range(self.max_substeps):
            response = self.vlm.chat(messages=messages, tools=AGENTIC_NAV_TOOLS, max_tokens=1536)
            t_calls = response.tool_calls

            # If no tool call returned, use resilient fallback
            if not t_calls:
                action_out = NavAction(
                    dx=self.max_range_m,
                    dy=0.0,
                    dz=0.0,
                    yaw_delta=0.0,
                    u=0.5,
                    v=0.5,
                    is_stop=False,
                    reasoning=response.content or "Proceeding forward along primary axis",
                    tool_calls=[{"name": "fallback_move", "u": 0.5, "v": 0.5}]
                )
                break

            primary_call = t_calls[0]
            fn_name = primary_call.get("function", {}).get("name", "move_to")
            args = primary_call.get("function", {}).get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            recorded_tool_calls.append({"substep": substep, "name": fn_name, "args": args})

            # Handle query_depth
            if fn_name == "query_depth":
                points = args.get("points", [{"u": 0.5, "v": 0.5}])
                depth_results = []
                for pt in points:
                    u_val = float(pt.get("u", 0.5))
                    v_val = float(pt.get("v", 0.5))
                    d_m = query_metric_depth_patch(observation.depth_map, u_val, v_val)
                    depth_results.append({"u": u_val, "v": v_val, "depth_m": round(d_m, 2)})

                messages.append({
                    "role": "assistant",
                    "content": response.content,
                    "tool_calls": [primary_call]
                })
                messages.append({
                    "role": "tool",
                    "name": "query_depth",
                    "content": json.dumps({"status": "success", "results": depth_results})
                })
                continue

            # Handle stop_arrival
            elif fn_name == "stop_arrival":
                u_val = float(args.get("u", 0.5))
                v_val = float(args.get("v", 0.5))
                reason = str(args.get("reason", "Destination landmark reached"))
                d_m = query_metric_depth_patch(observation.depth_map, u_val, v_val)
                action_out = NavAction(
                    u=u_val,
                    v=v_val,
                    depth_m=d_m,
                    is_stop=True,
                    reasoning=reason,
                    tool_calls=recorded_tool_calls,
                    latency_s=response.latency_s
                )
                break

            # Handle move_to
            else:
                u_val = float(args.get("u", 0.5))
                v_val = float(args.get("v", 0.5))
                reason = str(args.get("reason", "Proceeding toward visual waypoint"))
                d_m = query_metric_depth_patch(observation.depth_map, u_val, v_val, default_depth=self.max_range_m)

                dx, dy, dz, yaw_offset = backproject_pixel_to_body_frame(
                    u=u_val,
                    v=v_val,
                    depth_m=d_m,
                    intrinsics=self.intrinsics,
                    max_range_m=self.max_range_m
                )

                action_out = NavAction(
                    dx=dx,
                    dy=dy,
                    dz=dz,
                    yaw_delta=yaw_offset,
                    u=u_val,
                    v=v_val,
                    depth_m=d_m,
                    is_stop=False,
                    reasoning=reason,
                    tool_calls=recorded_tool_calls,
                    latency_s=response.latency_s
                )
                break

        if action_out is None:
            action_out = NavAction(
                dx=self.max_range_m,
                u=0.5,
                v=0.5,
                is_stop=False,
                reasoning="Default step",
                tool_calls=recorded_tool_calls
            )

        self.memory_history.append({"step_id": observation.step_id, "action": action_out})
        return action_out
