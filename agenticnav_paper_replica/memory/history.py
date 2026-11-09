"""Action-reasoning history manager for AgenticNav.

Maintains a sliding 6-step queue of past actions, rationales, and execution outcomes.
Supports ablation modes:
    - full: Full action-reasoning history
    - reasoning_to_task_progress: Replaces model rationale with coarse task progress text
    - without_history: Suppresses history entirely
"""

from collections import deque
from dataclasses import dataclass, field
import json
from typing import Deque, Dict, List, Optional, Union

from agenticnav_paper_replica.geometry.pinhole import VIEW_LABELS


@dataclass
class HistoryRecord:
    step_idx: int
    from_node: int
    to_node: int
    action_type: str  # "move_to", "recall", "query_depth", "stop"
    action_params: dict
    reason: str
    outcome: Optional[dict] = None
    task_progress_summary: Optional[str] = None


class ActionHistory:
    """Sliding window history buffer of length max_history (default 6)."""

    def __init__(self, max_history: int = 6, mode: str = "full"):
        self.max_history = max_history
        self.mode = mode  # "full", "reasoning_to_task_progress", "without_history"
        self._history: Deque[HistoryRecord] = deque(maxlen=max_history)

    def append(
        self,
        step_idx: int,
        from_node: int,
        to_node: int,
        action_type: str,
        action_params: dict,
        reason: str,
        outcome: Optional[dict] = None,
        task_progress_summary: Optional[str] = None,
    ) -> None:
        rec = HistoryRecord(
            step_idx=step_idx,
            from_node=from_node,
            to_node=to_node,
            action_type=action_type,
            action_params=action_params,
            reason=reason,
            outcome=outcome,
            task_progress_summary=task_progress_summary,
        )
        self._history.append(rec)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    def get_records(self) -> List[HistoryRecord]:
        return list(self._history)

    def format_for_prompt(self) -> str:
        """Formats the history queue into prompt text for the VLM."""
        if self.mode == "without_history" or len(self._history) == 0:
            return "No previous actions recorded."

        lines: List[str] = []
        for rec in self._history:
            # Format action call
            act_type = rec.action_type
            params = rec.action_params
            if act_type == "move_to":
                v_idx = params.get("view", 0)
                v_label = VIEW_LABELS.get(v_idx, f"View {v_idx}")
                u = params.get("u", 0.0)
                v = params.get("v", 0.0)
                act_str = f"move_to(view={v_label}, u={u:.2f}, v={v:.2f})"
            elif act_type == "recall":
                act_str = f"recall(node={params.get('node')}, view={params.get('view')})"
            elif act_type == "query_depth":
                pts = params.get("points", [])
                act_str = f"query_depth({len(pts)} points)"
            elif act_type == "stop":
                act_str = "stop()"
            else:
                act_str = f"{act_type}({json.dumps(params)})"

            # Format outcome
            out_str = ""
            if rec.outcome:
                status = rec.outcome.get("status", "")
                if status == "executed":
                    cmd_rho = rec.action_params.get("rho_m", rec.outcome.get("quantized_rho_m", 0.0))
                    actual_moved = rec.outcome.get("moved_m", cmd_rho)
                    collided = rec.outcome.get("collided", False)
                    th = rec.action_params.get("theta_deg", rec.outcome.get("quantized_theta_deg", 0.0))
                    fwd_steps = rec.action_params.get("forward_steps", 0)

                    if cmd_rho <= 0.0 or fwd_steps == 0:
                        out_str = f" -> Executed (ROTATED {th:+.0f}° in place, 0.00m forward; Note: View 0 now faces your new heading)"
                    elif collided or (actual_moved < 0.5 * cmd_rho):
                        out_str = f" -> Executed (displaced {actual_moved:.2f}m, turn {th:+.0f}° | PHYSICAL COLLISION: Contact with obstacle shortened displacement from {cmd_rho:.2f}m to {actual_moved:.2f}m; you are still near your previous position)"
                    else:
                        out_str = f" -> Executed (displaced {actual_moved:.2f}m, turn {th:+.0f}°)"
                elif status == "reselect":
                    cause = rec.outcome.get("cause", "unknown")
                    out_str = f" -> Reselect ({cause})"
                else:
                    out_str = f" -> {status}"

            # Format rationale / progress text
            if self.mode == "reasoning_to_task_progress":
                info_text = rec.task_progress_summary or f"Progressed from Node {rec.from_node} towards goal."
                lines.append(f"- Step {rec.step_idx} (Node {rec.from_node} -> Node {rec.to_node}): [Progress: {info_text}] [Action: {act_str}{out_str}]")
            else:
                # full mode
                lines.append(f"- Step {rec.step_idx} (Node {rec.from_node} -> Node {rec.to_node}): [Reason: {rec.reason}] [Action: {act_str}{out_str}]")

        return "\n".join(lines)

