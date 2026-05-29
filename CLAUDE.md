# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

## Build & Test

This project uses `uv`.

```bash
uv sync                          # Install dependencies
uv run python -m unittest        # Run all tests
uv run main.py run --method agentic   # Headless demo run
uv run main.py experiment        # Paired baseline-vs-agentic sweeps
uv run main.py ui                # Solara browser GUI (http://127.0.0.1:8765)
```

## Architecture Overview

A custom tick-based 2D simulator for evaluating decentralized UAV swarm decision
methods for a disaster-mapping paper. Methods (`static`, `rules`,
`task_consideration`, `agentic`) are swappable behind a single `SwarmMethod`
interface; the world model, events, communication, metrics, and renderer are
shared as an experiment harness.

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

- The `agentic` method is decentralized by design: per-UAV decisions from local
  observation, peer messages, and onboard state — no global oracle. Keep it that
  way (see the Decentralization Principle in the design spec).
- Changing only `method_name` must be enough to swap methods; no other subsystem
  should branch on the method.
