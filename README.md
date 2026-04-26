# Agentic UAV Swarm Simulator

Custom tick-based 2D simulator for evaluating decentralized UAV swarm decision methods for a disaster-mapping paper. The simulator keeps the world, movement, communication, event injection, metrics, and visualization shared across methods so the decision layer can be swapped cleanly.

## Methods

- `static`: pre-mission static partition baseline.
- `rules`: deterministic rule-adaptive baseline.
- `agentic`: structured agentic swarm method with dynamic role/target decisions.

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

Run all tests:

```bash
uv run python -m unittest
```

## GUI

The browser UI provides:

- method selection for `static`, `rules`, and `agentic`
- `Reset`, `Next Step`, and `End` controls
- a Mesa-style grid representation of sector state and UAV roles
- live metrics for coverage, active UAVs, and messages

## Architecture

- `agentic_uav/simulation.py`: grid world, UAV state, tick loop, communication, events, and metrics.
- `agentic_uav/methods.py`: swappable `SwarmMethod` implementations.
- `agentic_uav/scenarios.py`: reusable demo scenario construction.
- `agentic_uav/gui.py`: Solara GUI entrypoint.
- `agentic_uav/gui_support.py`: testable portrayal and metric-series helpers for the GUI.
- `agentic_uav/rendering.py`: static Matplotlib snapshot rendering.
- `main.py`: launcher for the GUI and headless run mode.

## Documentation

- Design spec: `docs/superpowers/specs/2026-04-26-swappable-uav-swarm-simulator-design.md`
- Next steps: `docs/NEXT_STEPS.md`

## Current Scope

The simulator is intentionally mission-level. It does not model low-level aerodynamics, flight control, or real UAV hardware. The paper-facing goal is to compare adaptation, coverage progress, communication behavior, and recovery under disruptions.
