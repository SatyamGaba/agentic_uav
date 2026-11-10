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

### 3-STAGE DECISION PROTOCOL (MANDATORY AT EVERY STEP):

#### STAGE 1: Mandatory Pre-Stride Arrival Check (Prevents Overshooting / Missed Stops)
Before planning any new `move_to` action, evaluate if your PREVIOUS move already placed you inside the final destination zone:
1. **Egocentric Frustum Shift on Arrival**:
   - When **approaching** a goal from 3m–6m away, it sits in the center of **View 0 (Front)**.
   - When you have **ARRIVED** at a goal (0.4m–2.0m away), it often shifts OUT of the center of View 0 to lower foreground or side views!
2. **Arrival Signatures by Goal Type**:
   - **Doorways / Archways / Entrances** (*"stop in the doorway/archway/entrance"*): You have arrived when the doorframes/arch jambs flank you immediately in **View 1 (Right)** and **View 3 (Left)** at `d < 1.3m`, while View 0 looks into the subsequent room. **STOP IMMEDIATELY.** Do NOT take another step forward into the room in View 0!
   - **Furniture / Fixtures** (*"stop in front of / next to the table, sofa, bed, stairs, island, toilet, hot tub"*): You have arrived when the target object is in the lower foreground (`v >= 0.65`) of View 0 OR directly beside you in View 1 / View 3 at a `query_depth` distance of **`0.5m <= d <= 2.2m`**. **STOP IMMEDIATELY.** Taking another step will overshoot or walk past the object.
   - **Rooms / Landings** (*"wait in the lounge / sitting area / top of stairs"*): Once you cross the threshold and the defining room features surround you within `1.5m–2.5m` in Views 0/1/3, **STOP IMMEDIATELY.** Do not walk to the far exit of the room.

#### STAGE 2: Counterfactual Dual-Hypothesis Gate (Prevents Premature Stops)
Decompose the instruction into **Ordered Transit Milestones ($C_1 \\dots C_{{n-1}}$)** and the **Terminal Goal ($C_n$)**.
Whenever you consider calling `stop()`, you MUST explicitly evaluate two competing hypotheses in your reasoning:
- **`H_ARRIVED` (True Terminal Arrival)**: Every intermediate milestone ($C_1 \\dots C_{{n-1}}$) has been physically traversed (not merely seen in the distance), AND the current camera views + depth (`0.5m <= d <= 2.2m`) match the exact terminal anchor $C_n$.
- **`H_PREMATURE` (Counterfactual / Decoy Hypothesis)**: I have only completed an intermediate step (e.g. climbed the stairs or entered the hallway), OR I am looking at the target room/object from afar (`d > 2.5m` through a doorway), OR this object is a decoy before the final turn.
- **Decision Rule**: Actively try to prove `H_PREMATURE` using all 4 views and `query_depth`. You may call `stop()` ONLY if `H_PREMATURE` is refuted with high confidence and the target anchor $C_n$ is verified within `<= 2.2m`.

#### STAGE 3: Anti-Loop & BEV Trajectory Discipline (Prevents Pacing Loops)
1. **BEV Trajectory Inspection**: Inspect the cyan path on your BEV map before selecting a `move_to` target. Do NOT place a `move_to` target on a location already traversed by your trajectory line unless dead-end backtracking is required.
2. **Local Pacing Prevention**: If you have remained within the same ~2m area for >= 2 steps because you are close to the target landmark, **call `stop()` instead of pacing around the landmark.**
3. **Pure-Rotation Orientation**: If action history reports you rotated in place, remember View 0 now faces your new heading.
4. **Verified Move Execution**: Select the EXACT `(view, u, v)` coordinates verified by `query_depth` that returned `"clear": true`. Never blindly shift coordinates.
5. **Never Repeat a Blocked Target**: If `move_to` returns `reselect` or `no_motion`, do NOT reissue the exact same `(view, u, v)` target. Choose a different view or floor point."""

SYSTEM_PROMPT = build_system_prompt(floor_grounding=True)


def build_step_user_prompt(
    instruction: str,
    current_node: int,
    history_text: str,
    accumulated_distance_m: float = 0.0,
    reselect_feedback: Optional[str] = None,
    loop_warning: Optional[str] = None,
) -> str:
    """Builds the structured user message for the current decision step."""
    msg = f"### Goal Navigation Instruction:\n\"{instruction}\"\n\n"
    msg += f"### Current State:\n"
    msg += f"- Active Decision Node: {current_node}\n"
    msg += f"- Accumulated Distance Traversed: {accumulated_distance_m:.2f} meters\n\n"

    msg += f"### Action History (Recent Steps):\n{history_text}\n\n"

    if loop_warning:
        msg += f"### Loop / Oscillation Warning:\n{loop_warning}\n\n"

    if reselect_feedback:
        msg += f"### Warning / Reselect Feedback:\n{reselect_feedback}\n\n"
        msg += "Please review the obstacle feedback and select an alternative unobstructed target or view.\n"
    else:
        msg += (
            "### Instructions for this Step (Follow 3-Stage Decision Protocol):\n"
            "1. Stage 1 (Pre-Stride Arrival Check): Did your previous move place you inside the terminal zone? "
            "Check Views 0, 1, and 3. If doorframes flank you in Views 1/3 (< 1.3m), or the target object is in foreground/flanking at 0.5m-2.2m, "
            "evaluate for arrival immediately!\n"
            "2. Stage 2 (Dual-Hypothesis Evaluation): If considering stop(), explicitly evaluate H_ARRIVED vs. H_PREMATURE. "
            "Have you physically traversed ALL intermediate milestones (C_1...C_n-1)? Is the target anchor <= 2.2m?\n"
            "3. Stage 3 (Motion & BEV Discipline): If continuing, check BEV map to avoid loops. Open transit uses standard strides (v in [0.65, 0.75]). "
            "Near-field approach uses micro-steps (v in [0.80, 0.90]). Query depth first and move to the exact verified point.\n"
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
