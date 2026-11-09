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
1. `query_depth(points)`: Inspect physical metric distance and 3D safety clearance for candidate targets in meters. Returns depth statistics (min, mean, median), 3D safety clearance (`clear`: true/false, `obstacle_cause`), and exact action preview (`preview_theta_deg`, `preview_rho_m`) showing the turn angle and displacement `move_to` will execute.
   * NOTE: Depth readings indicate local obstacle/wall clearance. A depth reading of 2.5m or 3.0m DOES NOT mean you have arrived at your destination! It simply indicates open floor or a wall at that distance.
2. `recall(node, view)`: Inspect past camera views from any previously visited node (1 <= node <= t-1) to re-verify landmarks or re-orient.
3. `move_to(view, u, v, reason)`: Navigate towards the point (u, v) seen in view. Select the EXACT (view, u, v) coordinates verified by `query_depth` that returned `"clear": true`.
4. `stop(reason)`: Declare arrival at the final goal destination.

### Tool Call Output Format:
You MUST invoke your chosen action using either native tool calling or formatted as a JSON code block:
```json
{{
  "name": "move_to",
  "arguments": {{
    "view": 0,
    "u": 0.5,
    "v": 0.65,
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

### MANDATORY TERMINAL & STEP CONTROL RULES:
1. **Milestone Progression: Destination vs. Transit Landmarks**:
   - Navigation instructions specify a sequence of milestones ending at a destination landmark.
   - Transit landmarks (e.g., "walk past the dining table and kitchen counter", "walk through the living room") are route waypoints along the journey. NEVER stop at intermediate transit landmarks or starting objects!
   - You MUST only call `stop` when you reach the FINAL destination landmark described in the concluding clause of the instruction.

2. **Peripheral Portal Check (Archway / Doorway / Room Entrance)**:
   - When the instruction says "stop in/at the archway/doorway/entrance", inspect View 1 (Right) and View 3 (Left) in addition to View 0.
   - If door frames, arch jambs, or walls flank you immediately on the left and right (< 0.8m depth), or if View 0 looks through the arch into the room ahead, you are ALREADY standing inside the portal! Call `stop` now! DO NOT take another forward step through into the next room!

3. **Sub-Goal Visibility Gating (Corners & Blocked Turns)**:
   - If a required turn is blocked at u=0.5 (e.g. facing a corner or wall edge), check wider angles u in {0.3, 0.7} or take a small forward step (0.5m - 1.0m) to clear the wall corner and align with the opening before abandoning the turn!

4. **Exact Query-to-Move Mapping (No Blind Shifting)**:
   - When commanding `move_to`, select the EXACT `(view, u, v)` coordinates that `query_depth` verified as `"clear": true`. Do NOT shift to an unverified coordinate (e.g. querying v=0.75 and commanding v=0.85), which risks hitting unverified obstacles.

5. **Transit Stride vs. Near-Field Terminal Approaching**:
   - Open room/corridor transit: For clear distances >= 3.0m, select v in [0.65, 0.75] to cover ground with standard 1.5m - 2.5m strides.
   - Near-field terminal approach: When approaching within 2.5m of the final destination landmark, select near-field points (v in [0.80, 0.90]) to execute precise micro-steps (0.5m - 1.0m) and position accurately without overshooting.

6. **Physical Collision Feedback**:
   - Pay close attention to action history. If your previous move reported a physical collision or shortened displacement (e.g. moved 0.14m of commanded 1.0m), you are still near your previous position! Adjust your heading or select an unobstructed angle.

7. **BEV Direction Check & Pure Rotations**:
   - If action history reports you rotated in place, remember that View 0 now faces your new heading.
   - Before taking a 180° turn (selecting View 2), inspect the cyan path on your BEV map. If turning around causes you to retrace steps backward along your cyan line into an already-visited room, DO NOT move forward. Rotate to View 1 or View 3 to find an orthogonal unexplored corridor.

8. **Depth Is Not Arrival**: A depth reading only measures local clearance to a surface. It does not prove that you have reached the instructed destination.

9. **Never Repeat a Blocked Target**: If `move_to` returns `reselect` or `no_motion`, do NOT reissue the exact same `(view, u, v)` target. Choose a different view or a different floor point."""

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
        msg += "Please review the obstacle feedback and select an alternative unobstructed target or view.\n"
    else:
        msg += (
            "### Instructions for this Step:\n"
            "1. Milestone Progress: Check which instructed milestone was just completed. DO NOT stop at intermediate transit landmarks or starting objects; only stop at the final destination landmark.\n"
            "2. Peripheral Portal Check: If stopping at an archway/doorway/entrance, check View 1 (Right) and View 3 (Left). If door frames flank you on left and right, or View 0 looks through into the subsequent room, you are already inside the portal—call `stop` now!\n"
            "3. Landmark Approach & Stride: During open transit (depth >= 3.0m), use standard strides (v in [0.65, 0.75]). Only use micro-steps (v in [0.80, 0.90]) when approaching within 2.5m of the final destination.\n"
            "4. Verified Move Execution: Select the EXACT (view, u, v) target verified by `query_depth` that returned `clear: true`.\n"
            "5. BEV & Heading Check: If your previous move was a pure in-place turn, View 0 now faces your new heading. Avoid retracing backward along your past cyan path on the BEV map.\n"
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
    """Builds the final warning turn before a safety deadlock (neutral, no arrival hints)."""
    target_text = f"(view={target_view}, u={target_u:.2f}, v={target_v:.2f})"
    if repeats >= 2:
        failure_text = (
            f"Your target {target_text} has failed {repeats} time(s) with cause "
            f"`{last_cause}`. Repeating it will trigger a safety deadlock."
        )
    else:
        attempt_count = attempts if attempts is not None else repeats
        failure_text = (
            f"You have made {attempt_count} consecutive failed move attempts. The most recent "
            f"target {target_text} failed with cause `{last_cause}`. Another failed target will "
            "trigger a safety deadlock."
        )

    return (
        "### OBSTACLE SAFETY WARNING\n"
        f"You are at decision node {current_node} after {accumulated_distance_m:.2f} m of travel.\n"
        f"{failure_text}\n\n"
        f"Goal instruction: \"{instruction}\"\n\n"
        "Inspect the current camera views and BEV map to choose your next action:\n"
        "- `move_to(view, u, v, reason)`: Choose an alternative open floor point or view to advance along an unobstructed passage.\n"
        "- `stop(reason)`: Invoke only if you have independently verified that you have reached the instructed destination landmark.\n"
        "Do not repeat the failed obstructed target."
    )
