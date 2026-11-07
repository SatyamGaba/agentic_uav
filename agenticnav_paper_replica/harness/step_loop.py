"""Multi-turn decision step loop and evidence/reselect budgeting for AgenticNav (v7 Specification)."""

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from agenticnav_paper_replica.geometry.pinhole import parse_view_index
from agenticnav_paper_replica.harness.tools import get_available_tools
from agenticnav_paper_replica.harness.executor import ToolExecutor
from agenticnav_paper_replica.harness.prompts import (
    SYSTEM_PROMPT,
    build_system_prompt,
    build_step_user_prompt,
    build_arrival_verification_prompt,
)
from agenticnav_paper_replica.memory.history import ActionHistory


TOOL_NAME_ALIASES = {
    "move": "move_to",
    "navigate": "move_to",
    "move_forward": "move_to",
    "forward": "move_to",
    "goto": "move_to",
    "depth": "query_depth",
    "inspect_depth": "query_depth",
    "check_depth": "query_depth",
    "get_depth": "query_depth",
    "memory": "recall",
    "recall_node": "recall",
    "arrive": "stop",
    "finish": "stop",
    "halt": "stop",
}


def canonicalize_tool_name(name: str) -> str:
    if not name:
        return ""
    clean = str(name).strip().lower()
    return TOOL_NAME_ALIASES.get(clean, clean)


def count_images_in_messages(messages: List[Dict[str, Any]]) -> int:
    """Counts the total number of visual images across all messages in the request."""
    cnt = 0
    for m in messages:
        if "rgb_views" in m and m["rgb_views"]:
            cnt += len(m["rgb_views"])
        if "bev_image" in m and m["bev_image"] is not None:
            cnt += 1
        if "recalled_image" in m and m["recalled_image"] is not None:
            cnt += 1
        if "images" in m and m["images"]:
            cnt += len(m["images"])
    return cnt


def estimate_tokens_in_messages(messages: List[Dict[str, Any]], image_token_weight: int = 512) -> int:
    """Estimates the total token count of a request (text characters / 3.5 + image tokens)."""
    text_chars = 0
    image_count = count_images_in_messages(messages)
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            text_chars += len(c)
        elif isinstance(c, list):
            for part in c:
                if isinstance(part, dict) and "text" in part:
                    text_chars += len(part["text"])
    estimated_text_tokens = int(math.ceil(text_chars / 3.5))
    estimated_image_tokens = image_count * image_token_weight
    return estimated_text_tokens + estimated_image_tokens


@dataclass
class StepLoopResult:
    action_type: str  # "move_to", "stop", "deadlock", "protocol_violation", "error"
    action_payload: Dict[str, Any]
    evidence_query_count: int
    consecutive_reselect_count: int
    malformed_call_count: int = 0
    terminated_deadlock: bool = False
    terminated_protocol_violation: bool = False
    arrival_verification_triggered: bool = False
    turn_log: List[Dict[str, Any]] = field(default_factory=list)


class StepDecisionLoop:
    """Executes the multi-turn reasoning and tool invocation loop for a single decision step t."""

    def __init__(
        self,
        executor: ToolExecutor,
        max_evidence_queries: int = 4,   # E_max [P]
        max_consecutive_reselects: int = 5,  # R_max [I]
        max_malformed_calls: int = 3,    # M_max [I]
        enable_depth: bool = True,
        enable_recall: bool = True,
        enable_arrival_verification: bool = True,
        floor_grounding_prompt: bool = True,
    ):
        self.executor = executor
        self.max_evidence_queries = max_evidence_queries
        self.max_consecutive_reselects = max_consecutive_reselects
        self.max_malformed_calls = max_malformed_calls
        self.enable_depth = enable_depth
        self.enable_recall = enable_recall
        self.enable_arrival_verification = enable_arrival_verification
        self.floor_grounding_prompt = floor_grounding_prompt

    def run_step(
        self,
        instruction: str,
        current_node: int,
        history: ActionHistory,
        rgb_views: Dict[int, np.ndarray],
        depth_views: Dict[int, np.ndarray],
        obstacle_points_agent_xz: np.ndarray,
        bev_image: Optional[np.ndarray],
        vlm_client_fn: Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], Dict[str, Any]],
        initial_reselect_count: int = 0,
        accumulated_distance_m: float = 0.0,
    ) -> StepLoopResult:
        """Runs the multi-turn loop for decision step t (Blueprint v7 Section 6)."""
        evidence_count = 0
        consecutive_reselects = initial_reselect_count
        malformed_count = 0
        turn_log = []
        last_failed_target: Optional[Tuple[int, float, float]] = None
        duplicate_target_repeats = 0
        verification_triggered = False

        history_text = history.format_for_prompt()
        initial_user_prompt = build_step_user_prompt(
            instruction=instruction,
            current_node=current_node,
            history_text=history_text,
            accumulated_distance_m=accumulated_distance_m,
            reselect_feedback=None,
        )

        system_prompt = build_system_prompt(floor_grounding=self.floor_grounding_prompt)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": initial_user_prompt,
                "rgb_views": rgb_views,
                "bev_image": bev_image,
            },
        ]

        while True:
            # Check reselect deadlock budget
            if consecutive_reselects >= self.max_consecutive_reselects:
                return StepLoopResult(
                    action_type="deadlock",
                    action_payload={"cause": "max_consecutive_reselects_reached", "count": consecutive_reselects},
                    evidence_query_count=evidence_count,
                    consecutive_reselect_count=consecutive_reselects,
                    malformed_call_count=malformed_count,
                    terminated_deadlock=True,
                    arrival_verification_triggered=verification_triggered,
                    turn_log=turn_log,
                )

            # Check malformed call protocol violation budget
            if malformed_count >= self.max_malformed_calls:
                return StepLoopResult(
                    action_type="protocol_violation",
                    action_payload={"cause": "max_malformed_calls_reached", "count": malformed_count},
                    evidence_query_count=evidence_count,
                    consecutive_reselect_count=consecutive_reselects,
                    malformed_call_count=malformed_count,
                    terminated_protocol_violation=True,
                    arrival_verification_triggered=verification_triggered,
                    turn_log=turn_log,
                )

            # Determine available tools (strip evidence tools when E_max reached or verification is active)
            if verification_triggered:
                tools = get_available_tools(enable_depth=False, enable_recall=False)
            else:
                allow_depth = self.enable_depth and (evidence_count < self.max_evidence_queries)
                allow_recall = self.enable_recall and (evidence_count < self.max_evidence_queries)
                tools = get_available_tools(enable_depth=allow_depth, enable_recall=allow_recall)

            req_img_count = count_images_in_messages(messages)
            req_est_tokens = estimate_tokens_in_messages(messages)

            vlm_response = vlm_client_fn(messages, tools)
            turn_log.append({
                "role": "assistant",
                "response": vlm_response,
                "request_image_count": req_img_count,
                "estimated_tokens": req_est_tokens,
            })

            # Record assistant turn in messages for proper ChatML alternation
            assistant_thought = vlm_response.get("thought", "")
            assistant_tool_calls = vlm_response.get("tool_calls", [])
            assistant_entry: Dict[str, Any] = {
                "role": "assistant",
                "content": assistant_thought,
            }
            if assistant_tool_calls:
                assistant_entry["tool_calls"] = assistant_tool_calls
            messages.append(assistant_entry)

            tool_calls = assistant_tool_calls
            for tc in tool_calls:
                tc["name"] = canonicalize_tool_name(tc.get("name", ""))

            # Handle no tool calls (counts against M_max)
            if not tool_calls:
                malformed_count += 1
                reprompt_msg = {
                    "role": "user",
                    "content": (
                        "No valid navigation tool was invoked. You MUST choose and call one tool "
                        "(move_to, stop, query_depth, recall) by formatting your tool call as a JSON block, e.g.:\n"
                        "```json\n"
                        '{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.55, "reason": "Advance along corridor"}}\n'
                        "```\n"
                        "Output your reasoning and the JSON tool call."
                    ),
                }
                messages.append(reprompt_msg)
                turn_log.append(reprompt_msg)
                continue

            # Separate evidence vs terminal calls
            evidence_calls = [tc for tc in tool_calls if tc["name"] in ("query_depth", "recall")]
            terminal_calls = [tc for tc in tool_calls if tc["name"] in ("move_to", "stop")]

            # During arrival verification, evidence tools are closed.
            if evidence_calls and verification_triggered:
                malformed_count += 1
                verify_tool_msg = {
                    "role": "tool",
                    "name": evidence_calls[0]["name"],
                    "content": json.dumps({
                        "status": "unavailable",
                        "message": "Evidence tools are closed for this step. Call stop or a DIFFERENT move_to target.",
                    }),
                }
                messages.append(verify_tool_msg)
                turn_log.append(verify_tool_msg)
                continue

            # Case 1: Evidence calls present (and under evidence budget)
            if evidence_calls and (evidence_count < self.max_evidence_queries):
                evidence_count += 1
                # Execute evidence calls in order
                for tc in evidence_calls:
                    t_name = tc["name"]
                    t_args = tc.get("arguments", {})
                    if isinstance(t_args, str):
                        try:
                            t_args = json.loads(t_args)
                        except Exception:
                            malformed_count += 1
                            t_args = {}

                    if t_name == "query_depth":
                        exec_res = self.executor.execute_query_depth(
                            points=t_args.get("points", []),
                            depth_views=depth_views,
                        )
                    else:  # recall
                        exec_res = self.executor.execute_recall(
                            node=int(t_args.get("node", 1)),
                            view=parse_view_index(t_args.get("view", 0)),
                        )

                    tool_msg = {
                        "role": "tool",
                        "name": t_name,
                        "content": json.dumps(exec_res, default=str),
                    }
                    if t_name == "recall" and exec_res.get("status") == "success":
                        # Hard rule for recall: drop older recalled images to respect the request image cap
                        for prev_msg in messages:
                            if "recalled_image" in prev_msg:
                                prev_msg.pop("recalled_image", None)
                        tool_msg["recalled_image"] = exec_res.get("image")

                    messages.append(tool_msg)
                    turn_log.append(tool_msg)

                # If terminal calls were also issued in this turn, answer with deferred (Section 6.2)
                for tc in terminal_calls:
                    def_msg = {
                        "role": "tool",
                        "name": tc["name"],
                        "content": json.dumps({
                            "status": "deferred",
                            "message": "Evidence query executed first. Please inspect evidence and reissue your move decision."
                        }),
                    }
                    messages.append(def_msg)
                    turn_log.append(def_msg)

                # If evidence budget just reached, add prompt instruction
                if evidence_count >= self.max_evidence_queries:
                    limit_msg = {
                        "role": "user",
                        "content": "Evidence query budget reached for this step. A move_to or stop action is now required."
                    }
                    messages.append(limit_msg)
                    turn_log.append(limit_msg)

                continue

            # Case 2: Terminal calls only
            if terminal_calls:
                # If multiple terminal calls with no evidence, execute first, answer rest as ignored
                primary_tc = terminal_calls[0]
                ignored_calls = terminal_calls[1:]

                for ig_tc in ignored_calls:
                    ig_msg = {
                        "role": "tool",
                        "name": ig_tc["name"],
                        "content": json.dumps({
                            "status": "ignored",
                            "message": "Multiple motion actions in single turn. Only the primary action is evaluated."
                        }),
                    }
                    messages.append(ig_msg)
                    turn_log.append(ig_msg)

                t_name = primary_tc["name"]
                t_args = primary_tc.get("arguments", {})
                if isinstance(t_args, str):
                    try:
                        t_args = json.loads(t_args)
                    except Exception:
                        malformed_count += 1
                        t_args = {}

                if t_name == "stop":
                    return StepLoopResult(
                        action_type="stop",
                        action_payload=t_args,
                        evidence_query_count=evidence_count,
                        consecutive_reselect_count=0,
                        malformed_call_count=malformed_count,
                        arrival_verification_triggered=verification_triggered,
                        turn_log=turn_log,
                    )

                elif t_name == "move_to":
                    t_view = parse_view_index(t_args.get("view", 0))
                    t_u = float(t_args.get("u", 0.5))
                    t_v = float(t_args.get("v", 0.5))
                    exec_res = self.executor.execute_move_to(
                        view=t_view,
                        u=t_u,
                        v=t_v,
                        reason=str(t_args.get("reason", "")),
                        depth_views=depth_views,
                        obstacle_points_agent_xz=obstacle_points_agent_xz,
                    )

                    if exec_res.get("status") == "executed":
                        return StepLoopResult(
                            action_type="move_to",
                            action_payload=exec_res,
                            evidence_query_count=evidence_count,
                            consecutive_reselect_count=0,
                            malformed_call_count=malformed_count,
                            arrival_verification_triggered=verification_triggered,
                            turn_log=turn_log,
                        )
                    else:
                        cause = str(exec_res.get("cause", "reselect"))
                        target_key = (t_view, round(t_u, 3), round(t_v, 3))
                        is_duplicate = target_key == last_failed_target

                        if is_duplicate:
                            duplicate_target_repeats += 1
                        else:
                            last_failed_target = target_key
                            duplicate_target_repeats = 1

                        consecutive_reselects += 1

                        should_verify = False
                        if self.enable_arrival_verification and not verification_triggered:
                            if duplicate_target_repeats >= 2:
                                should_verify = True
                            elif consecutive_reselects >= max(1, self.max_consecutive_reselects - 1):
                                should_verify = True

                        if should_verify:
                            verification_triggered = True
                            verification_msg = {
                                "role": "user",
                                "content": build_arrival_verification_prompt(
                                    instruction=instruction,
                                    current_node=current_node,
                                    target_view=t_view,
                                    target_u=t_u,
                                    target_v=t_v,
                                    last_cause=cause,
                                    repeats=duplicate_target_repeats,
                                    accumulated_distance_m=accumulated_distance_m,
                                    attempts=consecutive_reselects,
                                ),
                                "arrival_verification": True,
                            }
                            messages.append(verification_msg)
                            turn_log.append(verification_msg)
                            continue

                        feedback_payload = dict(exec_res)
                        feedback_payload["target"] = {
                            "view": t_view,
                            "u": round(t_u, 3),
                            "v": round(t_v, 3),
                        }
                        feedback_payload["duplicate_target_repeats"] = duplicate_target_repeats
                        if cause == "no_motion":
                            feedback_payload["guidance"] = (
                                "Forward distance to target is less than 0.25m after standoff subtraction (too close or facing an obstacle). "
                                "Do NOT repeat this exact target. If the destination landmark is directly in front of you, invoke stop(). "
                                "Otherwise, select a different view (1: Right, 3: Left) or a different floor point to find an open path."
                            )
                        elif cause == "corridor_blocked":
                            feedback_payload["guidance"] = (
                                "Corridor to selected point is obstructed by an obstacle. "
                                "Choose a DIFFERENT unobstructed floor point or turn to an adjacent view. "
                                "Call stop() only if the instructed destination landmark is directly in front of you."
                            )
                        else:
                            feedback_payload["guidance"] = (
                                "The selected target failed. Choose a DIFFERENT view or floor point. "
                                "Do not repeat the same (view, u, v) target."
                            )
                        reselect_msg = {
                            "role": "tool",
                            "name": "move_to",
                            "content": json.dumps(feedback_payload),
                        }
                        messages.append(reselect_msg)
                        turn_log.append(reselect_msg)
                        continue

            # Fallback for unrecognized tools
            malformed_count += 1
            err_msg = {
                "role": "user",
                "content": f"Unrecognized tool calls: {[tc.get('name') for tc in tool_calls]}.",
            }
            messages.append(err_msg)
            turn_log.append(err_msg)
