# Event-triggered LLM cadence and a provider-agnostic decider adapter

## Status

accepted

## Decision

The `agentic` (LLM-driven) method is invoked at **mission start** and **only when
an event fires** (`dropout`, `urgent_sector`, `block_sector`) — **not every
tick**. Between events, UAVs mechanically execute the persisted plan (movement via
the existing `_move_toward` logic). The LLM sits behind a **provider-agnostic
seam**: an `AgentDecider` protocol with a single `decide(observation) -> action
JSON` method. The default implementation is a dependency-free `MockAgentDecider`;
no real API dependency is committed yet and the vendor is decided later. Decider
output is **strictly validated**, with a **greedy fallback** on any invalid
output. Strict run-to-run reproducibility is **not** required; the scenario `seed`
still pins the world.

## Context

Calling an LLM on every UAV every tick would be millions of calls per trial —
infeasible in cost and latency, and unnecessary because nothing changes most
ticks. Tying the cadence to events drops call volume to roughly **tens per
trial**: re-plan when the situation actually changes, then execute the plan.

Separately, we did not want to lock the project to a specific LLM vendor before
the experiment design is settled, nor add a heavyweight API dependency to a
harness that must run in tests and smoke runs. A thin adapter keeps the vendor
choice a late, low-cost decision.

## Considered options

- **Per-tick LLM calls.** Rejected: millions of calls per trial, no benefit since
  most ticks are uneventful.
- **Commit to one vendor's SDK now.** Rejected: premature lock-in; breaks
  offline tests/smoke runs; vendor choice is reversible and should stay so.
- **Event-triggered cadence + provider-agnostic `AgentDecider` with a mock**
  (chosen): cheap, testable offline, and a real vendor is a drop-in
  `AgentDecider` later (only `build_method` / env wiring changes).

## Consequences

- The decider seam is `decide(observation) -> action JSON`; the wrapper
  serializes only **local** observation fields (per ADR-0001 — no global coverage
  or urgent), calls the decider, validates the action JSON (in-bounds, not
  blocked, role in `{coverage, priority_responder, relay}`), and falls back to
  `greedy_decision` on any violation, malformed output, or exception.
- The default `MockAgentDecider` is deterministic and mirrors greedy, so under
  the mock **agentic behaves like greedy**. That is the intended scaffold state,
  not a result (see EXPERIMENTS.md).
- No `anthropic` / `openai` dependency is added in this pass; swapping in a real
  vendor changes only the decider construction.
- Reproducibility is intentionally relaxed: a real LLM may vary run-to-run. The
  scenario `seed` still fixes dropouts/urgent events identically for every method,
  preserving the fair comparison.
