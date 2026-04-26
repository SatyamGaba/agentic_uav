# Next Steps

This roadmap turns the current prototype into a paper-ready simulation and evaluation repo.

## 1. Stabilize The Simulator Core

- Add scenario configuration loading from files so experiments are not hardcoded in Python.
- Add deterministic seed handling for all random choices.
- Expand event types for UAV dropout, blocked sectors, urgent task insertion, visibility degradation, and communication blackout regions.
- Add explicit mission-success criteria: coverage target, deadline, urgent-sector completion, and failure recovery.

## 2. Improve Method Comparisons

- Make `static`, `rules`, and `agentic` report the same action traces.
- Strengthen the rule-based baseline with fixed but credible trigger logic.
- Add action validation for unsupported roles, blocked targets, invalid messages, and unsafe moves.
- Add role-transition metrics for `coverage`, `priority_responder`, and `relay`.

## 3. Build Experiment Sweeps

- Create a batch runner for repeated trials over method, seed, swarm size, and disruption severity.
- Save outputs to a structured `runs/` directory with one summary file per trial.
- Add experiment families for dropout, communication degradation, blocked sectors, and urgent task insertion.
- Generate aggregate tables for success rate, recovery time, coverage by deadline, and communication overhead.

## 4. Upgrade Visualization

- Add GUI controls for scenario family, swarm size, and seed.
- Add optional communication-link overlays on the grid.
- Add paper-ready plot exports for coverage over time, recovery time, success rate, and message overhead.
- Add an animation export path for one representative run.

## 5. Add LLM Agent Adapter

- Define a strict JSON input/output schema for LLM-backed decisions.
- Add a validator that maps invalid LLM output to a safe fallback action.
- Start with an API-backed adapter for a small qualitative demo.
- Keep the main quantitative experiments reproducible with the structured `agentic` method.

## 6. Paper Readiness

- Align metrics with the paper claims: faster recovery, higher mission completion, and graceful degradation under communication/failure stress.
- Add scripts that reproduce every table and figure from saved run outputs.
- Document assumptions and limitations in the repo so the paper methods section can reference them directly.
- Keep the simulation framing clear: mission-level reasoning only, not low-level UAV control.

## Suggested Immediate Order

1. Add file-based scenario configs.
2. Add batch experiment runner and CSV/JSON outputs.
3. Implement the four disruption experiment families.
4. Add aggregate metrics and paper plots.
5. Add the LLM JSON adapter as a separate optional backend.
