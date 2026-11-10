# AGENT REMEDIATION BRIEF: Closing the 51% OSR -> 50% SR Gap (Post-Arm O)

## 0. Context & Ground Truth Metrics (Job #649 - Arm O)
- **Current Metrics:** OSR = 51.0% | SR = 21.0% | SPL = 13.32 | Safety Deadlocks = 8 | Protocol Violations = 1
- **Core Bottleneck:** 30 episodes reach the <= 3.0m goal zone (51 OSR vs 21 SR) but fail to stop properly:
  - `STOP_AFTER_REACH` (18 eps): Reached 0.39m–2.69m from goal, failed to recognize arrival, took 2–4 extra strides into the next room, and stopped late (3.27m–14.95m).
  - `MISSED_STOP` (11 eps): 11/11 (100%) physically reached <= 3.0m (0.32m–2.73m), then wandered until step 26 without calling `stop()`.
  - `STOP_PREMATURE` (26 eps): Stopped at 3.71m–20.33m due to confirmation bias on intermediate milestones.
  - `BUDGET_EXCEEDED` (15 eps): 9 episodes stuck in local pacing/oscillation loops; 6 episodes took wrong macro-branches.
  - `SAFETY_DEADLOCK` (8 eps): 32/32 reselects caused by `invalid_depth` (d <= 0.50m near-plane clipping in `pinhole.py`), NOT obstacle collisions.
  - `PROTOCOL_VIOLATION` (1 ep, `ep_0018`): Malformed non-JSON output and invalid `view` arguments.

---

## Task 1: Prompt Overhaul — Dual-Hypothesis Verification & Frustum Arrival Semantics

Update the VLM system/navigation prompt to enforce a **3-Stage Decision Protocol** at every step:

### Stage 1: Mandatory Pre-Stride Arrival Check (Fixes `STOP_AFTER_REACH` & `MISSED_STOP`)
Before planning any new `move_to` action, evaluate if your PREVIOUS move already placed you inside the final destination zone ($C_n$):
1. **Egocentric Frustum Shift on Arrival:**
   - When you are **approaching** a goal from 3m–6m away, it sits in the center of **View 0 (Front)**.
   - When you have **ARRIVED** at a goal (0.4m–2.0m away), it often shifts OUT of the center of View 0!
2. **Arrival Signatures by Goal Type:**
   - **Doorways / Archways / Entrances** (*"stop in the doorway/archway/entrance"*): You have arrived when the doorframes/arch pillars are immediately beside you in **View 1 (Left)** and **View 3 (Right)** at `d < 1.3m`, while View 0 looks into the next room. **STOP IMMEDIATELY.** Do NOT step forward into the room visible in View 0.
   - **Furniture / Fixtures** (*"stop in front of / next to the table, sofa, bed, stairs, island, toilet, hot tub"*): You have arrived when the target object is in the lower foreground (`v >= 0.65`) of View 0 OR directly beside you in View 1 / View 3 at a `query_depth` distance of **`0.5m <= d <= 2.2m`**. **STOP IMMEDIATELY.** Taking another step will overshoot or walk past the object.
   - **Rooms / Landings** (*"wait in the lounge / sitting area / top of stairs"*): Once you cross the threshold and the defining room features surround you within `1.5m–2.5m` in Views 0/1/3, **STOP IMMEDIATELY.** Do not walk to the far exit of the room.

### Stage 2: Counterfactual Dual-Hypothesis Gate (Fixes `STOP_PREMATURE`)
Decompose the instruction into **Ordered Transit Milestones ($C_1 \dots C_{n-1}$)** and the **Terminal Goal ($C_n$)**.
Whenever you consider calling `stop()`, you MUST explicitly evaluate two competing hypotheses in your reasoning:
- **`H_ARRIVED` (True Terminal Arrival):** Every intermediate milestone ($C_1 \dots C_{n-1}$) has been physically traversed (not merely seen in the distance), AND the current camera views + depth (`0.5m <= d <= 2.2m`) match the exact terminal anchor $C_n$.
- **`H_PREMATURE` (Counterfactual / Decoy Hypothesis):** I have only completed an intermediate step (e.g., climbed the stairs or entered the hallway), OR I am looking at the target room/object from afar (`d > 2.5m` through a doorway), OR this object is a decoy before the final turn.
- **Decision Rule:** Actively try to prove `H_PREMATURE` using all 4 views and `query_depth`. You may call `stop()` ONLY if `H_PREMATURE` is refuted with high confidence and the target anchor $C_n$ is verified within `<= 2.2m`.

### Stage 3: Anti-Loop & BEV Trajectory Discipline (Fixes `BUDGET_EXCEEDED`)
- Inspect the BEV map trajectory before selecting a `move_to` target.
- Do NOT place a `move_to` target on a location already traversed by your trajectory line unless dead-end backtracking is required.
- If you have remained within the same ~2m area for >= 2 steps because you are close to the target landmark, **call `stop()` instead of pacing around the landmark.**

---

## Task 2: Harness Fixes for `SAFETY_DEADLOCK` (`pinhole.py` & Reselect Handler)

All 8 deadlocks (`ep_0013, 0014, 0015, 0045, 0058, 0060, 0031, 0038`) occurred because `convert_raw_depth()` rejected pixels with `d <= 0.50m` (`invalid_depth`) before NavMesh pathfinding was invoked. Implement the following 3 fixes:

1. **Upward Vertical Ray Search on Near-Clip (`pinhole.py`):**
   - When `move_to(view, u, v)` reads a raw depth `0 < d <= d_min` (near-plane clip on close table, wall, or stair riser), scan upward along the same column `u` (`v_candidate` from `v` down to `0.35` in increments of `-0.04`) to find the first valid depth `d > d_min` in the same horizontal direction, and deproject that point to the NavMesh.
2. **Azimuth Ground-Projection Fallback for Actuation:**
   - If the entire local patch around `(u, v)` in `move_to` clips at `d <= 0.50m`, compute the horizontal camera ray bearing for `(view, u)`, project a waypoint `1.25m` forward along that heading at floor elevation, and pass it to `pathfinder.snap_point()`. Allow Habitat's Oracle NavMesh follower to route around the close table (`ep_0013-0015`) or up the stair incline (`ep_0058, 0060, 0031`).
3. **Informative Reselect Diagnostic + Wall-Arrival Hint (`ep_0038`):**
   - When returning an `invalid_depth` reselect to the model, differentiate `near_clip (d <= 0.50m)` from `far_clip`:
     `"NEAR_PLANE_CLIP (d < 0.50m): You are standing < 0.50m from a surface/wall/table in this view. (1) If you have already reached the goal location, call stop() now! (2) Do NOT make tiny adjustments to (u, v) around the same spot; either look higher (v in [0.35, 0.50]) or pick an open direction in another view (View 1, 2, or 3)."`

---

## Task 3: Harness Fix for `PROTOCOL_VIOLATION` (`ep_0018`)

Eliminate tool-call formatting failures to guarantee `PROTOCOL_VIOLATION == 0`:
1. Enable strict structured output / JSON schema enforcement on tool calls (`view` enum `[0, 1, 2, 3]`, float `u, v` clamped/validated in `[0.0, 1.0]`).
2. Add a pre-parser sanitizer that extracts the valid JSON object from surrounding text/markdown and normalizes string view indices (`"0"` -> `0`, `"View 1"` -> `1`) before counting a retry fault.
3. On retry, append the exact schema error string so the model self-corrects immediately.

---

## Task 4: Loop vs. Exploration Audit & Validation Protocol

1. **BEV / Trajectory Loop Audit Script (`audit_arm_oracle_loops.py`):**
   - For the 15 `BUDGET_EXCEEDED` episodes, compute the step-to-step revisitation matrix `D[i, j] = ||pos_i - pos_j||_2` for `|i - j| >= 2`.
   - Classify each episode as `OSCILLATION_LOOP` (if `>= 4` steps revisit a prior pose within `1.5m`) vs. `TOPOLOGICAL_DIVERGENCE` (wrong branch).
   - Add an automatic `[LOOP DETECTED]` flag to the runtime prompt whenever runtime `||pos_t - pos_{t-k}||_2 < 1.2m` for `k >= 2`.
2. **Targeted 20-Episode Smoke Test Before Full 100-Episode Job:**
   - Run a fast validation batch on 20 high-signal failure episodes from Arm O:
     - 6 `STOP_AFTER_REACH`: `ep_0010, ep_0012, ep_0023, ep_0036, ep_0007, ep_0026`
     - 4 `MISSED_STOP`: `ep_0039, ep_0049, ep_0054, ep_0071`
     - 5 `STOP_PREMATURE`: `ep_0017, ep_0005, ep_0016, ep_0040, ep_0042`
     - 4 `SAFETY_DEADLOCK`: `ep_0013, ep_0058, ep_0031, ep_0038`
     - 1 `PROTOCOL_VIOLATION`: `ep_0018`
- **Success Gate for Smoke Test:** 0 safety deadlocks, 0 protocol violations, and >= 65% SR on the `STOP_AFTER_REACH` + `MISSED_STOP` subset before launching the full 100-episode benchmark.

