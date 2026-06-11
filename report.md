# Agentic UAV — Repository Analysis (Refined)

> **Verdict:** The simulation backbone is solid — tick loop, swappable methods, experiment sweeps, GUI, and 73 passing tests. But the codebase implements almost none of the per-UAV architecture described in the [framework spec](file:///c:/Users/Kruti/Downloads/agentic_uav-main/agentic_uav-main/reference/agentic_uav_swarm_framework_and_experiments.md). The current code is a flat target-selection loop; the spec describes a layered sense→think→act agent with a local world model, health monitor, communication manager, tool interfaces, and safety governor. Until the `AgenticMethod` embodies this architecture, the paper cannot claim to evaluate an "onboard mission-level agent."

---

## 1. What Works Well

| Area | Assessment |
|---|---|
| **Tick-loop simulation** | Clean `Simulation.step()` cycle: events → deliver messages → observe → decide → resolve → sense → log. |
| **Swappable methods** | `SwarmMethod` protocol + `build_method()` factory. Swapping with a single config string is the right pattern. |
| **Three baseline implementations** | `static`, `rules`, and `task_consideration` cover the spec's Baselines A, B, and C. |
| **Experiment harness** | Paired seeded sweeps across swarm sizes × dropout fractions × all 4 methods. CSV/JSON/plot export. |
| **GUI** | Solara-based grid visualisation with paths, comm links, metric charts, event timeline. Functional and polished. |
| **Tests** | 73 tests pass, covering simulation backbone, all methods, communication, rendering, CLI, GUI support, config persistence. |
| **Communication model** | Range-limited Manhattan-distance delivery with TTL-based urgent forwarding. Correct foundation. |

---

## 2. Framework Spec vs. Implementation — Architecture Gap

The [framework spec §3–§5](file:///c:/Users/Kruti/Downloads/agentic_uav-main/agentic_uav-main/reference/agentic_uav_swarm_framework_and_experiments.md) describes a two-level system. Each UAV hosts an internal decision stack (Level 1) and the swarm coordinates peer-to-peer (Level 2). Here is exactly what exists vs. what the spec requires:

### Level 1: Per-UAV Internal Architecture

| Spec Component (§4) | Required Function | Current Code | Status |
|---|---|---|---|
| **§4.1 Mission spec** | Machine-readable mission goal, constraints, priorities, completion criteria | `ScenarioConfig.mission_type` — a string (`"survey"` or `"disaster_mapping"`) | ⚠️ Stub — no structured goal, no priority weights, no completion criteria per UAV |
| **§4.2 Perception & state estimation** | Structured summaries from sensors, perception confidence, hazard detection, event flags | `ObservationBuilder` produces `{self, urgent_cells, nearby, messages}` | ⚠️ Partial — no perception confidence, no visibility degradation, no hazard detection events |
| **§4.3 Communication manager** | Message packaging, priority routing, link-quality awareness, peer-event triggers | `NetworkModel` + `Message` — flat broadcast within range | ⚠️ Partial — no link quality filtering, no priority routing, no comm-event triggers |
| **§4.4 Health & resource monitor** | Energy level, sensor health, degradation status, availability level | `UavState.energy=1.0` and `UavState.health="nominal"` — **never read or consumed** | ❌ Not implemented |
| **§4.5 Local world model** | Per-UAV belief state, team estimate, known hazard map, task status, uncertainty | Does not exist. Methods read the global `simulation.world.sectors` directly. | ❌ Not implemented |
| **§4.6 Agent core** | Sense→Think→Act reasoning: role decisions, task bids, replan triggers, tool invocations, confidence estimates | `AgenticMethod.decide_tick()` — 15 lines: nearest-urgent or nearest-uncovered | ❌ Not implemented (simplest of all 4 methods) |
| **§4.7 Tool interfaces** | Task allocation module, route planner, mission progress checker, hazard assessment, contingency planner | Does not exist. Agent directly picks a cell. | ❌ Not implemented |
| **§4.8 Safety governor** | Validates proposed actions, applies geofence rules, provides safe fallback | Does not exist. `resolve_actions()` does basic bounds-check only. | ❌ Not implemented |
| **§4.9 Autopilot/execution** | Waypoint tracking, execution feedback | `_move_toward()` — 1 Manhattan step per tick, no feedback | ⚠️ Minimal (adequate for simulation abstraction) |

### Level 2: Inter-UAV Coordination

| Spec Component (§6) | Required Messages | Current Code | Status |
|---|---|---|---|
| **Heartbeat & status** | ID, role, position, health, energy, comm quality | Not emitted by `agentic`. `rules` emits `intent_summary` on heartbeat interval. | ⚠️ Partial |
| **Task bid / commitment** | Task ID, suitability score, feasibility, commitment state | `task_consideration` emits `task_commitment`. `agentic` does not. | ❌ Missing from agentic |
| **Intent summary** | Objective, near-term plan, role, expected contribution | `agentic` emits `intent_summary` but **never reads incoming ones** | ⚠️ Emit-only, no consumption |
| **Hazard alert** | Hazard type, location, urgency, timestamp, confidence | Not implemented by any method | ❌ Not implemented |
| **Failure/degradation notice** | UAV unavailable, role abandonment, sensor degradation, low-energy | Not implemented by any method | ❌ Not implemented |
| **Observation/coverage update** | Region covered, blocked area, environmental state | Not implemented by any method | ❌ Not implemented |

### Reasoning Loop (§5)

| Spec Component | Required | Current | Status |
|---|---|---|---|
| **Periodic trigger** | Low-rate reassessment cycle independent of events | All methods run every tick — no distinction between periodic and event-driven | ❌ No trigger mechanism |
| **Event-driven trigger** | Immediate invocation on hazard, peer failure, connectivity change, task completion | `handle_event()` exists in protocol but is **never called** by the simulation | ❌ Dead code |
| **Sense phase** | Aggregate local world model + comms + health into context summary | `ObservationBuilder` creates a flat dict, no fusion or summarisation | ⚠️ Minimal |
| **Think phase** | Deliberative reasoning: role assessment, coordination check, tool selection, replan decision | `AgenticMethod` does a single `if urgent → respond, else → cover` | ❌ No deliberation |
| **Act phase** | Structured decision package: task bid, role change, planner call, comms, contingency | `Action(target_cell, new_role, messages)` — correct structure but agentic only fills target+role | ⚠️ Structure exists, underused |

---

## 3. The "Agentic" Method Is the Simplest Implementation

This is the single most damaging problem. The paper's contribution is the agentic architecture, but:

```
AgenticMethod.decide_tick()     →  ~15 lines  (nearest-urgent fallback nearest-uncovered)
RuleAdaptiveMethod.decide_tick() →  ~50 lines  (urgent tracking, peer conflict avoidance, patrol)
TaskConsiderationMethod          →  ~55 lines  (utility scoring, conflict penalty, hysteresis, commitment)
```

- **No intent ingestion**: Emits `intent_summary` messages but never reads inbox. No peer awareness.
- **No conflict avoidance**: Multiple agentic UAVs will converge on the same urgent cell.
- **No hysteresis**: No task commitment tracking — switches targets every tick.
- **No relay role**: The spec and GUI legend describe `relay` UAVs, but no method ever assigns it.
- **No health-aware reasoning**: Never reads `energy` or `health`.
- **No event-driven replanning**: No `handle_event()` invocation.

**In any fair experiment, `task_consideration` will outperform `agentic`.** The paper's core claim cannot be supported.

---

## 4. Simulation Fidelity Gaps

### 4.1 Dead data model fields

| Field | Declared | Read by anything? |
|---|---|---|
| `Sector.hazard` | `models.py` | ❌ |
| `Sector.visibility` | `models.py` | ❌ |
| `Sector.comm_quality` | `models.py` | ❌ |
| `UavState.energy` | `models.py` | ❌ (set to 1.0 forever) |
| `UavState.outbox` | `models.py` | ❌ (messages go via `NetworkModel.enqueue`) |

### 4.2 All UAVs start at `(0, 0)`

`_uav_start_cells()` returns `[(0, 0)] * uav_count`. The spec says "common base or nearby sector" — this should be a small cluster, not identical coordinates.

### 4.3 No energy drain or return-to-base

The energy field exists but is never decremented. No UAV ever runs low on energy or needs to return to base. This eliminates an entire class of interesting allocation decisions.

### 4.4 No communication degradation

The spec requires "intermittent packet loss" and "localized communication blackout" (§9.4). `NetworkModel` always delivers perfectly within range. The `comm_quality` field is never used.

### 4.5 No perception degradation

The spec requires "partial visibility degradation" (§9.4). `Sector.visibility` is never read; sensing radius is uniform everywhere.

### 4.6 Binary coverage

Coverage is 0.0 or 1.0. No partial accumulation, no decay, no revisit requirement. This reduces coverage to a counting problem.

### 4.7 Greedy 1-step movement

UAVs take one Manhattan step toward their target per tick. No pathfinding around blocked cells, no energy cost for movement, no speed variation.

---

## 5. Experiment & Metrics Gaps

### 5.1 Only 2 of 4 experiments implementable

| Experiment (spec §9.7) | Implementable now? |
|---|---|
| **Exp 1: UAV dropout recovery** | ⚠️ Partially — dropout events exist but no per-UAV recovery metrics |
| **Exp 2: Communication degradation** | ❌ No comm degradation mechanics |
| **Exp 3: Dynamic task insertion** | ⚠️ Partially — 1 hardcoded urgent event, not configurable |
| **Exp 4: Perception degradation** | ❌ No visibility mechanics |

### 5.2 Missing metrics (spec §9.6)

| Metric | Status |
|---|---|
| Adaptability (time to first reassignment, recovery cycles) | ❌ |
| Task completion time (per-task, not just global) | ❌ |
| Role reallocation effectiveness (abandoned tasks recovered) | ❌ |
| Resilience under comm loss (success vs. packet loss) | ❌ |
| Decision latency (event-to-decision delay) | ❌ |
| Task distribution imbalance | ❌ |
| Local decision ratio | ❌ |
| Per-UAV coverage contribution | ❌ |
| Coordination overhead (messages per coverage unit) | ❌ |

### 5.3 No action traces

Only aggregate metrics are saved. No per-tick, per-UAV decision log for post-hoc analysis or debugging.

### 5.4 No statistical significance

Only `mean` and `sem`. No confidence intervals, no significance tests, no effect sizes.

### 5.5 Minimal plot suite

Only 3 types: dropout-vs-coverage, dropout-vs-success, representative timeline. Need recovery curves, radar charts, box plots with significance markers.

---

## 6. Code Quality Issues

| Issue | Detail |
|---|---|
| **Loose typing** | Methods accept `simulation: object` instead of `Simulation` — breaks IDE, type checker, readability |
| **No logging** | Uses `print()` only. Need `logging` module for research reproducibility. |
| **Bare `pyproject.toml`** | `description = "Add your description here"`. No author, license, dev deps. |
| **No linting/type-checking** | No mypy, ruff, or CI configuration. |
| **CSS in Python** | 470 lines of CSS embedded as a Python string in `gui.py`. |
| **Hardcoded `/tmp` path** | `rendering.py` sets `MPLCONFIGDIR=/tmp/matplotlib` — fails on Windows. |
| **No docstrings** | Zero docstrings across all modules. |

---

## 7. Test Coverage Assessment

| Module | Coverage | Gap |
|---|---|---|
| `simulation.py` | ✅ Good | — |
| `policy.py` (static, rules, task_consideration) | ✅ Good | Edge cases for multi-UAV conflict |
| `policy.py` (agentic) | ⚠️ 2 tests | No test for message reading (because it doesn't read messages) |
| `communication.py` | ✅ Via integration | No isolated `NetworkModel` unit tests |
| `experiments.py` | ⚠️ Basic | `ticks=3`, no plot generation coverage |
| `models.py` | ❌ None | `coverage_ratio`, `manhattan`, `neighborhood` untested in isolation |
| `planning.py` | ❌ None | `nearest_uncovered`, `nearest_open_urgent`, `ObservationBuilder` untested |
| Integration | ❌ None | No end-to-end experiment sweep test |

---

## 8. Summary

The simulation harness (tick loop, method swapping, experiment runner, GUI) is **well built and functional**. The baselines (`static`, `rules`, `task_consideration`) are reasonable implementations.

But the project fails at its primary objective: **the `AgenticMethod` does not implement the framework described in the reference spec**. It doesn't have a local world model, health monitoring, communication management, tool interfaces, safety governance, or deliberative reasoning. It's a 15-line nearest-cell selector — simpler than all three baselines.

The repo is at approximately **35% completion** for research paper readiness. The architecture is clean enough for incremental improvement without a rewrite, but the agentic method and simulation fidelity features need substantial work.
