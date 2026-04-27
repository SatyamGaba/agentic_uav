# Next Steps

This roadmap turns the current prototype into a paper-ready simulation and evaluation repo.

## 1. Stabilize The Simulator Core

- Add scenario configuration loading from files so experiments are not hardcoded in Python.
- Add deterministic seed handling for all random choices.
- Expand event types for UAV dropout, blocked sectors, urgent task insertion, visibility degradation, and communication blackout regions.
- Add explicit mission-success criteria: coverage target, deadline, urgent-sector completion, and failure recovery.

## 2. Improve Method Comparisons

- Make `static`, `rules`, `task_consideration`, and `agentic` report the same action traces.
- Strengthen the rule-based baseline with fixed but credible trigger logic.
- Implement `task_consideration` as the modern non-agentic decentralized scheduling baseline inspired by Chen, Li, and Peng (2023).
- Add action validation for unsupported roles, blocked targets, invalid messages, and unsafe moves.
- Add role-transition metrics for `coverage`, `priority_responder`, and `relay`.

## 3. Build Experiment Sweeps

- Extend the batch runner beyond `survey_dropout` and `disaster_urgent_dropout` once communication degradation is implemented.
- Save richer action traces alongside the current trial and aggregate outputs.
- Add communication degradation / blackout experiment families.
- Expand aggregate tables with network fragmentation once packet loss or blackout regions exist.

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

## 7. Decentralization Fidelity

- Document the exact information boundary for each UAV: local observation, onboard state, current assignment, and received peer messages.
- Add metrics for local decision ratio, peer-message dependence, and any action that required global synchronization.
- Keep `static` as a centralized/pre-mission comparison baseline, while ensuring `rules`, `task_consideration`, and `agentic` are described and evaluated as decentralized runtime methods.
- Avoid implementation changes that let the `agentic` method use hidden global state except through explicit observations or range-limited delayed messages.

## Suggested Immediate Order

1. Add file-based scenario configs.
2. Add communication degradation and blackout event support.
3. Extend experiment sweeps to communication stress.
4. Add richer action traces and task reassignment metrics.
5. Document and enforce the decentralized information boundary.
6. Add the LLM JSON adapter as a separate optional backend.
