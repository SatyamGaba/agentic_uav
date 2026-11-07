"""Prompts and context formatting for AgenticNav (High-Density Context & Milestone-Guided)."""

from typing import Dict, List, Optional
from agenticnav_paper_replica.geometry.pinhole import VIEW_LABELS


def build_system_prompt(floor_grounding: bool = True) -> str:
    floor_text = ""
    if floor_grounding:
        floor_text = """   - The horizon / eye-level is at v = 0.50.
   - To navigate along open floor surfaces, select coordinates in the lower region of the view (typically v in [0.65, 0.85]).
   - Open doorways or distant corridor openings can be targeted near v in [0.50, 0.65].
"""
    return f"""You are an autonomous robot navigation agent. Your objective is to follow natural language navigation instructions to reach the target destination in an indoor environment.

### Perception and Sensor Setup:
At each decision step, you are provided with:
1. **Four panoramic camera views**:
   - View 0: Front (0°)
   - View 1: Right (-90°)
   - View 2: Back (180°)
   - View 3: Left (+90°)
   Normalized image coordinates (u, v) are in [0.0, 1.0], where (0.0, 0.0) is top-left and (1.0, 1.0) is bottom-right.
{floor_text}2. **Bird's-Eye View (BEV) Map**:
   - White: Explored free space / navigable corridors.
   - Black: Confirmed obstacles and walls.
   - Gray: Unknown / unexplored space.
   - Cyan/Blue line: Your past trajectory.
   - Amber numbered circles: Past decision nodes (Node 1, 2, ...).
   - Red circle with directional arrows: Your current position and heading (Node t).

### Available Tools:
1. `query_depth(points)`: Inspect physical metric distance and 3D clearance for candidate targets in meters. Returns depth statistics (min, mean, median) and action preview (`preview_theta_deg`, `preview_rho_m`) showing the turn angle and displacement `move_to` would execute.
   * NOTE: Depth readings indicate local obstacle/wall clearance. A depth reading of 2.5m or 3.0m DOES NOT mean you have arrived at your destination! It simply indicates open floor or a wall at that distance.
2. `recall(node, view)`: Inspect past camera views from any previously visited node (1 <= node <= t-1) to re-verify landmarks or re-orient.
3. `move_to(view, u, v, reason)`: Navigate towards the point (u, v) seen in view. Select clear floor space, doorways, or open corridors to advance along the route.
4. `stop(reason)`: Declare arrival at the final goal destination.

### Tool Call Output Format:
You MUST invoke your chosen action using either native tool calling or formatted as a JSON code block:
```json
{{
  "name": "move_to",
  "arguments": {{
    "view": 0,
    "u": 0.5,
    "v": 0.55,
    "reason": "Advance forward down corridor"
  }}
}}
```
Or for final destination arrival:
```json
{{
  "name": "stop",
  "arguments": {{
    "reason": "All milestones fulfilled; positioned directly in front of target landmark"
  }}
}}
```

### MANDATORY NAVIGATION & STOPPING RULES:
- **Instruction Milestones**: Break the instruction down into sequential milestones (e.g., 1. Exit room -> 2. Turn right into hallway -> 3. Walk towards landmark -> 4. Stop near landmark). Always track which milestone you are currently executing.
- **Maintain Forward Momentum**: As long as intermediate milestones remain, continue executing `move_to` along the open corridor, doorway, or floor towards the next waypoint.
- **Never Repeat a Blocked Target**: If `move_to` returns `reselect` or `no_motion`, do NOT reissue the exact same `(view, u, v)` target. Choose a different view or a different floor point.
- **Depth Is Not Arrival**: A depth reading only measures clearance to a surface. It does not prove that you have reached the instructed destination.
- **Goal Landmark Arrival**: When you have fulfilled all milestones and reached the target location (or are positioned directly facing/next to the destination landmark specified in the instruction), invoke `stop`.
- **Blocked by Landmark**: If forward motion is blocked because you have arrived directly in front of the goal object (e.g., table, counter, bed, or door specified in the instruction), call `stop`. If it is an unrelated obstacle on the way, turn to navigate around it."""

SYSTEM_PROMPT = build_system_prompt(floor_grounding=True)


def build_step_user_prompt(
    instruction: str,
    current_node: int,
    history_text: str,
    accumulated_distance_m: float = 0.0,
    reselect_feedback: Optional[str] = None,
) -> str:
    """Builds the structured user message for the current decision step."""
    msg = f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
    msg += f"### Current State:\n"
    msg += f"- Active Decision Node: {current_node}\n"
    msg += f"- Accumulated Distance Traversed: {accumulated_distance_m:.2f} meters\n\n"

    msg += f"### Action History (Recent Steps):\n{history_text}\n\n"

    if reselect_feedback:
        msg += f"### Warning / Reselect Feedback:\n{reselect_feedback}\n\n"
        msg += "Please review the obstacle feedback. If you have arrived at your target landmark, call `stop`. Otherwise, select an alternative open direction.\n"
    else:
        msg += (
            "### Instructions for this Step:\n"
            "1. Deconstruct the goal instruction into sequential milestones.\n"
            "2. Identify which milestone was just completed and which milestone is next.\n"
            "3. If intermediate milestones remain, execute `move_to` along the corridor/doorway to progress.\n"
            "4. If all instruction milestones are fulfilled and you are at the target destination, call `stop`.\n"
            "Decide on your next action now."
        )

    return msg


def build_arrival_verification_prompt(
    instruction: str,
    current_node: int,
    target_view: int,
    target_u: float,
    target_v: float,
    last_cause: str,
    repeats: int,
    accumulated_distance_m: float = 0.0,
    attempts: Optional[int] = None,
) -> str:
    """Builds the one-shot terminal verification turn before a safety deadlock."""
    target_text = f"(view={target_view}, u={target_u:.2f}, v={target_v:.2f})"
    if repeats >= 2:
        failure_text = (
            f"Your target {target_text} has failed {repeats} time(s) with cause "
            f"`{last_cause}`. Repeating it will terminate the episode."
        )
    else:
        attempt_count = attempts if attempts is not None else repeats
        failure_text = (
            f"You have made {attempt_count} consecutive failed move attempts. The most recent "
            f"target {target_text} failed with cause `{last_cause}`. Another failed target will "
            "terminate the episode."
        )

    return (
        "### ARRIVAL VERIFICATION REQUIRED\n"
        f"You are at decision node {current_node} after {accumulated_distance_m:.2f} m of travel.\n"
        f"{failure_text}\n\n"
        f"Original instruction: \"{instruction}\"\n\n"
        "Inspect the current camera views one final time and choose exactly one action:\n"
        "1. `stop(reason)` if the instructed destination landmark is visible and you are directly next to it.\n"
        "2. `move_to(view, u, v, reason)` only if you select a DIFFERENT view or a DIFFERENT floor point.\n\n"
        "Do not repeat the failed target. Do not call stop merely because the path is blocked; "
        "stop only when the destination landmark itself matches the instruction."
    )
