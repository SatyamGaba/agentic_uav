# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Build & Test

This project uses `uv`.

```bash
uv sync                          # Install dependencies
uv run python -m unittest        # Run all tests
uv run main.py run --method agentic   # Headless demo run (proposed LLM method, mock decider)
uv run main.py run --method greedy    # Headless demo run (greedy heuristic baseline)
uv run main.py experiment        # Paired baselines-vs-agentic sweeps
uv run main.py ui                # Solara browser GUI (http://127.0.0.1:8765)
```

## Architecture Overview

A custom tick-based 2D simulator for evaluating decentralized UAV swarm decision
methods for a disaster-mapping paper. There are 5 methods: four heuristic
baselines (`static`, `rules`, `task_consideration`, `greedy`) and the proposed
LLM-driven method (`agentic`). All five are swappable behind a single
`SwarmMethod` interface; the world model, events, communication, metrics, and
renderer are shared as an experiment harness.

`greedy` is the renamed former `AgenticMethod` heuristic — it is a baseline, not
the contribution. `agentic` is the proposed method and the LLM seam: an
LLM-driven, decentralized decision agent whose thesis is that it beats the
heuristic baselines on recovery after disruption. All five methods receive the
**same** hybrid observation (static grid layout global; coverage/urgent
discovered locally via a per-UAV belief or learned through range-limited peer
messages), so they differ only in reasoning. The `agentic` LLM is
event-triggered and sits behind a provider-agnostic adapter (deterministic mock
decider as the default; no vendor API dependency yet), falling back to the
`greedy` heuristic on invalid output.

- `agentic_uav/simulation.py` — grid world, tick loop, events, metrics.
- `agentic_uav/policy.py` — swappable `SwarmMethod` implementations.
- `agentic_uav/planning.py` — `Action`, `MethodState`, `ObservationBuilder`.
- `agentic_uav/models.py` — core dataclasses (`Sector`, `UavState`, `WorldState`).
- `agentic_uav/communication.py` — range-limited, delayed `NetworkModel`.
- `agentic_uav/scenarios.py` — reusable demo scenario construction.
- `agentic_uav/experiments.py` — seeded sweeps, aggregates, paper plots.
- `agentic_uav/gui.py` / `gui_support.py` — Solara GUI.
- `agentic_uav/rendering.py` — static Matplotlib snapshots.

See `README.md`, `docs/NEXT_STEPS.md`, and
`docs/superpowers/specs/2026-04-26-swappable-uav-swarm-simulator-design.md`.

## Conventions & Patterns

- The `agentic` method (and every other method) is decentralized by design:
  per-UAV decisions from local observation, per-UAV belief, peer messages, and
  onboard state — no global oracle. Keep it that way (see the Decentralization
  Principle in the design spec).
- Observation symmetry is non-negotiable: all five methods get the same
  per-UAV observation (layout global; coverage/urgent discovered locally). They
  differ only in reasoning, never in what they can see.
- Changing only `method_name` must be enough to swap methods; no other subsystem
  should branch on the method. This invariant holds across all five methods,
  including the LLM `agentic` method.
- `handle_event()` is now invoked: `Simulation.step` calls it for each fired
  event, so a method can mark a re-plan as pending. Baselines stay no-ops; the
  `agentic` method uses it as the event-triggered re-plan hook (mission start
  plus dropout / urgent_sector / block_sector events).
- The `intent_summary` negotiation channel is now consumed: the `agentic`
  method ingests peer `intent_summary` messages as negotiation priors (and still
  emits its own) so colliding proposals deconflict synchronously within a tick.
- Metrics now carry action/plan traces: the `MetricsLogger` records replan and
  plan-change traces (e.g. `replan_count`, `plan_change_count`) used for the
  recovery-after-disruption analysis.
