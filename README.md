# Agentic UAV Swarm Simulator

Custom tick-based 2D simulator for evaluating decentralized UAV swarm decision methods for a disaster-mapping paper. The proposed `agentic` method is decentralized by design: each UAV makes mission-level role, target, and message decisions from its own observation, peer messages, and onboard state rather than from a central runtime controller. The simulator keeps the world, movement, communication, event injection, metrics, and visualization shared across methods only as an experiment harness so the decision layer can be swapped cleanly and compared fairly.

## Methods

- `static`: Baseline A, centralized pre-mission static partition for comparison.
- `rules`: Baseline B, decentralized deterministic rule-adaptive planner.
- `task_consideration`: Baseline C, decentralized task-consideration scheduler inspired by Chen, Li, and Peng (2023).
- `agentic`: proposed decentralized mission-level agentic planner with per-UAV dynamic role/target/message decisions.

Current code supports `static`, `rules`, `task_consideration`, and `agentic`.

## Quick Start

Install dependencies and use the project with `uv`:

```bash
uv sync
```

Launch the Mesa-style browser GUI:

```bash
uv run main.py
```

The GUI opens a Solara app on `http://127.0.0.1:8765` by default. If that port is busy, run:

```bash
uv run main.py ui --port 8766
```

Run the demo scenario headlessly:

```bash
uv run main.py run --method agentic --snapshot /tmp/agentic-uav-snapshot.png
```

Run the paired baseline-vs-agentic experiment sweeps:

```bash
uv run main.py experiment --output-dir runs/experiments
```

For a quick smoke run:

```bash
uv run main.py experiment --seed-count 2 --swarm-sizes 5 --dropout-fractions 0,0.25 --ticks 80 --no-plots
```

Run all tests:

```bash
uv run python -m unittest
```

## GUI

The browser UI provides:

- method selection for `static`, `rules`, `task_consideration`, and `agentic`
- `Reset`, `Next Step`, and `End` controls
- a Mesa-style grid representation of sector state and UAV roles
- live metrics for coverage, active UAVs, and messages

## Architecture

- `agentic_uav/simulation.py`: grid world, UAV state, tick loop, communication, events, and metrics.
- `agentic_uav/policy.py`: swappable `SwarmMethod` implementations.
- `agentic_uav/methods.py`: compatibility/export surface for method classes and helpers.
- `agentic_uav/scenarios.py`: reusable demo scenario construction.
- `agentic_uav/experiments.py`: paired seeded sweeps, aggregate metrics, and paper plot exports.
- `agentic_uav/gui.py`: Solara GUI entrypoint.
- `agentic_uav/gui_support.py`: testable portrayal and metric-series helpers for the GUI.
- `agentic_uav/rendering.py`: static Matplotlib snapshot rendering.
- `main.py`: launcher for the GUI and headless run mode.

## Documentation

- Design spec: `docs/superpowers/specs/2026-04-26-swappable-uav-swarm-simulator-design.md`
- Next steps: `docs/NEXT_STEPS.md`

## Current Scope

The simulator is intentionally mission-level. It does not model low-level aerodynamics, flight control, or real UAV hardware. The paper-facing goal is to compare adaptation, coverage progress, communication behavior, and recovery under disruptions while keeping the proposed runtime architecture decentralized.

The shared simulator state is not meant to imply that the `agentic` method has a global oracle or central mission controller. It is the evaluation engine. Agentic decisions should be based on explicit per-UAV observations, local state, and range-limited delayed messages.
