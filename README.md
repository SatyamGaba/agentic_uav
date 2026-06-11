# Agentic UAV Swarm Simulator

Custom tick-based 2D simulator for evaluating decentralized UAV swarm decision methods for a disaster-mapping paper. The proposed `agentic` method is an LLM-driven, decentralized decision agent: each UAV makes mission-level role, target, and message decisions from its own observation, peer messages, and onboard state rather than from a central runtime controller. The thesis is that this LLM-driven agent **beats the heuristic baselines on recovery after disruption**. The four heuristic baselines (`static`, `rules`, `task_consideration`, `greedy`) and the proposed `agentic` method all share the **same observation** so they differ only in reasoning. The simulator keeps the world, movement, communication, event injection, metrics, and visualization shared across methods only as an experiment harness so the decision layer can be swapped cleanly and compared fairly.

## Methods

The four heuristic baselines are `static`, `rules`, `task_consideration`, and `greedy`; the proposed LLM method is `agentic`.

- `static`: Baseline A, centralized pre-mission static partition for comparison.
- `rules`: Baseline B, decentralized deterministic rule-adaptive planner.
- `task_consideration`: Baseline C, decentralized task-consideration scheduler inspired by Chen, Li, and Peng (2023).
- `greedy`: Baseline D, decentralized greedy heuristic with per-UAV dynamic role/target/message decisions (formerly mislabeled "the proposed agentic method"; it is now correctly framed as a heuristic baseline).
- `agentic`: the proposed method. An LLM-driven, decentralized decision agent. It is event-triggered (it re-plans at mission start and when an event fires) and sits behind a provider-agnostic adapter that emits validated JSON actions, with a deterministic **mock** decider as the default for tests and smoke runs and no vendor API dependency added yet. On invalid LLM output it falls back to the `greedy` heuristic.

Current code supports `static`, `rules`, `task_consideration`, `greedy`, and `agentic`.

All five methods receive the same hybrid observation: the static grid layout is **global** (UAVs "have a map"), while dynamic state (coverage progress and urgent events) must be **discovered locally** through sensing or learned via range-limited peer messages. The methods differ only in how they reason over that shared observation.

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

Run the demo scenario headlessly (the proposed LLM method uses the mock decider by default):

```bash
uv run main.py run --method agentic --snapshot /tmp/agentic-uav-snapshot.png
```

Run a heuristic baseline headlessly (e.g. the `greedy` baseline):

```bash
uv run main.py run --method greedy --snapshot /tmp/greedy-uav-snapshot.png
```

Run the paired baselines-vs-agentic experiment sweeps:

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

- method selection for `static`, `rules`, `task_consideration`, `greedy`, and `agentic`
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

The simulator is intentionally mission-level. It does not model low-level aerodynamics, flight control, or real UAV hardware. The paper-facing goal is to compare adaptation, coverage progress, communication behavior, and recovery under disruptions, with the headline claim that the LLM-driven `agentic` method beats the heuristic baselines on recovery after disruption while keeping the proposed runtime architecture decentralized.

The shared simulator state is not meant to imply that any method has a global oracle or central mission controller. It is the evaluation engine. Every method, including the LLM-driven `agentic` method, decides from explicit per-UAV observations, local belief, and range-limited delayed messages. The `agentic` LLM is event-triggered and sits behind a provider-agnostic adapter (a deterministic mock decider by default), so swapping in a real vendor changes only the adapter, not the harness.
