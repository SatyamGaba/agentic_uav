# Next Steps

This roadmap turns the current prototype into a paper-ready simulation and evaluation repo.

> **Status update (2026-05-28).** Several foundational items are now DONE: the
> file-based decisions are captured as ADRs under `docs/adr/`; the decentralization
> boundary is enforced via a per-UAV belief plus observation symmetry (all five
> methods get the same hybrid observation — layout global, coverage/urgent
> discovered locally); `handle_event()` is wired into the tick loop; the
> `intent_summary` negotiation channel is consumed; action/plan traces are
> recorded in metrics; and the LLM adapter **seam exists with a deterministic
> mock** decider. The former `AgenticMethod` heuristic is now the `greedy`
> baseline, and `agentic` is the LLM-driven proposed method. Done items are
> marked inline below.

## 1. Stabilize The Simulator Core

- Add scenario configuration loading from files so experiments are not hardcoded in Python.
- Add deterministic seed handling for all random choices.
- Expand event types for UAV dropout, blocked sectors, urgent task insertion, visibility degradation, and communication blackout regions.
- Add explicit mission-success criteria: coverage target, deadline, urgent-sector completion, and failure recovery.

## 2. Improve Method Comparisons

- **DONE.** Make all five methods (`static`, `rules`, `task_consideration`, `greedy`, `agentic`) report the same action traces. Metrics now carry replan/plan-change traces shared across methods.
- Strengthen the rule-based baseline with fixed but credible trigger logic.
- Implement `task_consideration` as the modern non-agentic decentralized scheduling baseline inspired by Chen, Li, and Peng (2023).
- Add action validation for unsupported roles, blocked targets, invalid messages, and unsafe moves.
- Add role-transition metrics for `coverage`, `priority_responder`, and `relay`.

> Note: the heuristic formerly called the proposed agentic method is now the `greedy` baseline. The proposed `agentic` method is the LLM-driven, decentralized agent whose thesis is that it beats the four heuristic baselines (`static`, `rules`, `task_consideration`, `greedy`) on recovery after disruption.

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

## 5. Add LLM Agent Adapter (partially delivered)

The provider-agnostic adapter **seam is now in place**: an `AgentDecider`-style
`decide(observation) -> action JSON` interface with a deterministic, dependency-free
**mock** decider as the default, behind which a real vendor is a drop-in. The
proposed `agentic` method already uses this seam and is event-triggered. What
remains is wiring a real vendor and running the full quantitative evaluation.

- **DONE.** Define a strict JSON input/output schema for LLM-backed decisions.
- **DONE.** Add a validator that maps invalid LLM output to a safe fallback action (falls back to the `greedy` heuristic).
- **DONE (mock).** Provide a deterministic mock decider for tests and smoke runs with no vendor API dependency.
- **REMAINING.** Add a real vendor-backed adapter behind the same seam for a small qualitative demo.
- **REMAINING.** Run the full quantitative `agentic`-vs-baselines evaluation (the mock makes `agentic` mirror `greedy`, so the current smoke result is not yet a finding).
- Accept LLM run-to-run variation; the scenario `seed` still pins the world so the comparison stays fair.

## 6. Paper Readiness

- Align metrics with the paper claims: the LLM-driven `agentic` method achieves faster recovery after disruption, higher mission completion, and more graceful degradation under communication/failure stress than the heuristic baselines.
- Add scripts that reproduce every table and figure from saved run outputs.
- Document assumptions and limitations in the repo so the paper methods section can reference them directly.
- Keep the simulation framing clear: mission-level reasoning only, not low-level UAV control.

## 7. Decentralization Fidelity

- **DONE.** Document the exact information boundary for each UAV: it is hybrid and symmetric across all methods — static grid layout is global, while coverage progress and urgent events must be discovered locally (sensed nearby) or learned via range-limited delayed peer messages, held in a per-UAV belief. See `docs/adr/`.
- Add metrics for local decision ratio, peer-message dependence, and any action that required global synchronization.
- Keep `static` as a centralized/pre-mission comparison baseline, while ensuring `rules`, `task_consideration`, `greedy`, and `agentic` are described and evaluated as decentralized runtime methods.
- **DONE.** Enforce observation symmetry: no method reads hidden global dynamic state; every method (including the LLM `agentic` method) decides only from its per-UAV belief, local observation, and range-limited delayed messages.

## Suggested Immediate Order

1. Add file-based scenario configs.
2. Add communication degradation and blackout event support.
3. Extend experiment sweeps to communication stress.
4. ~~Add richer action traces and task reassignment metrics.~~ (action/plan traces DONE; task-reassignment metrics remain)
5. ~~Document and enforce the decentralized information boundary.~~ (DONE via per-UAV belief + observation symmetry; see `docs/adr/`)
6. ~~Add the LLM JSON adapter as a separate optional backend.~~ (seam + mock DONE; real vendor backend remains)
7. Wire a real vendor behind the `agentic` adapter seam and run the full `agentic`-vs-baselines recovery evaluation.
