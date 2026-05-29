# Decentralization and observation symmetry

## Status

accepted

## Decision

Every decision method — the four baselines (`static`, `rules`,
`task_consideration`, `greedy`) and the proposed `agentic` method — receives the
**identical observation** for a given UAV and decides from that UAV's own local
view only. There is no central planner. The static grid **layout** (dimensions,
blocked sectors, in-bounds) is global ("UAVs have a map"), but all **dynamic
state** — per-sector coverage progress and which sectors are urgent — is **local**:
each UAV accumulates it in a per-UAV `BeliefMap` learned by sensing its
neighborhood and by range-limited peer messages. Ground truth still lives in
`WorldState` and is written centrally for sensing and metrics, but **no decision
path reads global dynamic state**.

## Context

The pre-existing code was an unfair mix. `ObservationBuilder.build` injected a
global `urgent_cells` list (every urgent sector, regardless of where the UAV
was), and the planning helpers `nearest_open_urgent` / `nearest_uncovered`
scanned the entire `world.sectors` dictionary directly. That gave every method a
global oracle over dynamic state, which made the agentic-vs-baseline comparison
**invalid**: any "agentic" advantage could just be the oracle, not the reasoning.
The whole point of the harness is a fair comparison, so the leak had to go for
all methods at once.

## Considered options

- **Keep the global urgent list / full-map scans.** Rejected: it is the fairness
  leak; it makes the headline claim unfalsifiable.
- **Store belief on `UavState`.** Rejected: that bakes a decision-method concept
  into the world model and the GUI, breaking the "the simulator has no
  method-specific branch" invariant.
- **Per-UAV `BeliefMap` on `MethodState`** (chosen): decisions read belief,
  belief is updated from sensed neighborhood + received messages, and the world
  stays method-agnostic.

## Consequences

- Belief lives on `MethodState.beliefs: dict[str, BeliefMap]`, **not** on
  `UavState` — the world and GUI remain method-agnostic.
- `ObservationBuilder.build` gains a `beliefs` parameter and emits
  `known_urgent` from each UAV's belief; the global `urgent_cells` key is
  **removed**. Asserting `"urgent_cells"` is absent is a symmetry regression test.
- A UAV beyond sensing range with no peer message will **not** know about a far
  urgent sector — that is the correct, intended behavior, not a bug.
- Baseline numbers will change versus the old leaky harness. Treat the new
  numbers as correct: they reflect a swarm that must actually discover the world.
- Symmetry is enforced structurally: the observation is built once per UAV and
  handed to whatever method is running, so no method can see more than another.
