# Experiment Protocol

This document defines how the harness evaluates the proposed **agentic** (LLM)
method against the **baselines** (`static`, `rules`, `task_consideration`,
`greedy`). See `CONTEXT.md` for terminology and `docs/adr/` for the decisions this
protocol depends on (observation symmetry, in-tick negotiation, event-triggered
LLM cadence).

## Headline metric: recovery after disruption

The thesis is that the agentic method **recovers better after a disruption** than
the best baseline. We measure recovery two ways:

- **`recovery_slope`** — the rate of coverage progress in the ticks following a
  `dropout` event.
- **`recovery_ticks_to_target`** — how many ticks the swarm takes to re-reach the
  target coverage level after the disruption.

### MONOTONIC-COVERAGE caveat (read this)

Per-sector **coverage is monotonic** — it only ever increases; a UAV dropout does
not un-cover already-covered sectors. So `recovery_slope` does **not** measure
"regaining lost coverage." It measures **continued progress under reduced swarm
capacity** — how well the remaining UAVs keep covering ground after losing peers.
Report and label it that way. "Recovery" here = sustained forward progress
despite the loss, not clawing back something that decreased.

## Scenario

Use the focused **`disaster_urgent_dropout`** family: a disaster-mapping world in
which urgent sectors appear and a UAV dropout occurs mid-mission. This is the
scenario that directly exercises the headline metric — it forces the swarm to
re-plan and redistribute work after losing capacity, which is exactly where an
LLM agent is hypothesized to beat heuristics.

## Scale

Deliberately **small and targeted**, not the large factorial sweep:

- **Start with 1 seed and one swarm size** (e.g. the `recovery_focus_config`:
  `swarm_sizes=(6,)`, `family=disaster_urgent_dropout`, `ticks=120`,
  `grid_size=12`, no plots).
- The scenario `seed` pins the world — the same dropouts and urgent events are
  injected identically for every method, so the comparison is fair even with a
  single seed.
- **Optionally add a few more seeds later** purely as a variance check, once the
  single-seed signal looks worth chasing. Do not start there.

## Definition of "beats"

The agentic method **beats** the baselines when, against the **best** baseline on
the same pinned scenario, it:

1. sustains **faster post-disruption coverage progress** (`recovery_slope`),
   and/or **re-reaches target coverage sooner** (`recovery_ticks_to_target`),
2. **while staying competitive on overhead** — message volume, role-switch count,
   and replan count. A recovery win bought with a blow-up in communication or
   thrashing is not a clean win.

Secondary metrics (completion / success rate, total overhead) are reported for
context but are not the headline.

## Reproducibility

Strict run-to-run reproducibility is **not** required (see ADR-0003): a real LLM
may vary between runs. The scenario `seed` still pins the world — identical
dropouts and urgent events for every method — which is what keeps the comparison
fair. When using more than one seed, report variance across seeds rather than
expecting bit-identical runs.

## NOTE: smoke run is plumbing, not a result

The default decider is the `MockAgentDecider`, which mirrors greedy. **Under the
mock, the agentic method behaves like greedy** — this is the intended scaffold
state, not a finding. A smoke run therefore verifies *plumbing* (event-triggered
re-plan fires, negotiation resolves, traces and the new recovery columns are
written), **not** that agentic beats anything. A real-LLM evaluation is future
work; do not read the mock smoke numbers as a positive or negative result.

### Smoke command

```
uv run main.py experiment --family disaster_urgent_dropout --seed-count 1 --swarm-sizes 6 --ticks 120 --grid-size 12 --no-plots --output-dir runs/smoke
```

A successful smoke run completes and writes trial + aggregate output including the
`recovery_slope` and `recovery_ticks_to_target` columns.
