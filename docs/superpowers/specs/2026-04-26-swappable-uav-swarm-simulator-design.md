# Swappable UAV Swarm Simulator Design

## Summary

Build a custom Python, tick-based 2D disaster-mapping simulator with a sector grid as both the simulation state and the visualization layer. The architecture must make `Baseline A`, `Baseline B`, and the `Agentic` method directly swappable by changing only one top-level method selection, while keeping the world model, event schedule, communication model, metrics, seeds, renderer, and action validator identical.

The top-level swappable unit is a `SwarmMethod`, not only a per-UAV policy, because `Baseline A` needs a pre-mission global assignment step while the other methods can operate in a distributed runtime style.

## Core Architecture

- `ScenarioConfig`: immutable experiment description containing grid size, swarm size, mission horizon, event schedule, communication settings, and random seed.
- `WorldState`: square sector grid with per-sector coverage, priority, blockage, hazard, visibility quality, and communication quality.
- `UAVState`: per-UAV dynamic state with current cell, role, health, energy, target assignment, inbox, outbox, and active status.
- `Simulation`: shared tick loop with no method-specific branches.
- `ObservationBuilder`: builds the same structured observation format for every method.
- `NetworkModel`: neighbor-only communication with one-hop-per-tick delivery, range-limited direct links, and TTL-limited urgent propagation.
- `EventInjector`: introduces scheduled disruptions such as dropout, blocked sectors, degraded visibility, urgent-sector insertion, and communication blackouts.
- `MetricsLogger`: records tick-level and event-level metrics in a method-agnostic format.
- `Renderer`: produces snapshots and later animations/plots from simulator state and logs.

## Swappable Method Interface

All methods implement the same interface and return the same action objects:

```python
class SwarmMethod(Protocol):
    method_id: str

    def initialize_mission(self, simulation) -> MethodState:
        ...

    def decide_tick(self, simulation, observations, method_state) -> list[Action]:
        ...

    def handle_event(self, event, method_state) -> None:
        ...
```

The experiment runner swaps methods through `method_name` or the CLI flag `--method static|rules|agentic`. No other subsystem should change when the method changes.

Shared action schema:

- `continue_assignment`
- `retarget_sector`
- `switch_role`
- `handoff_sector`
- `send_message`
- `become_relay`

Shared role set:

- `coverage`
- `priority_responder`
- `relay`

## Method Definitions

- `Baseline A: StaticPartitionMethod`
  - Performs pre-mission grid partitioning.
  - Each UAV mostly follows its assigned sector list.
  - It does not perform meaningful dynamic reassignment when urgent sectors appear or a peer drops out.

- `Baseline B: RuleAdaptiveMethod`
  - Uses fixed triggers and heuristics at runtime.
  - It reacts to urgent sectors and nearby uncovered work through deterministic rules.
  - It does not perform broader mission reasoning or open-ended reprioritization.

- `Proposed: AgenticMethod`
  - Uses the same world, tools, movement, communication, and action schema.
  - Chooses role and target from structured observations, local mission state, and peer/message context.
  - May later be backed by a real LLM that emits the same validated JSON action package.

## Simulation Semantics

- One tick is one abstract mission-level decision step.
- Movement is cell-to-cell on the grid.
- Sensing uses a neighborhood footprint with radius `1`.
- Communication is local by default; no global instantaneous broadcast.
- Direct neighbor messages sent at tick `t` become visible at tick `t+1`.
- Multi-hop propagation takes multiple ticks through TTL-limited forwarding.
- Mission type is coverage-first disaster mapping with urgent-sector insertion.
- Mission success is measured by coverage progress, deadline completion, urgent response, and recovery after disruptions.

## Tick Loop

1. Inject scheduled events.
2. Clear stale inboxes and deliver one-hop messages.
3. Build observations for all active UAVs.
4. Call `decide_tick` on the selected `SwarmMethod`.
5. Validate and resolve actions through the shared simulation path.
6. Move UAVs and apply sensing coverage.
7. Enqueue outgoing messages for future ticks.
8. Log world state, actions, messages, and metrics.
9. Render or export state only outside method logic.

## Experiments

Run separate controlled studies, not one overloaded scenario:

- UAV dropout
- communication degradation / blackout regions
- blocked or hazardous sectors
- urgent-sector insertion

Recommended sweeps:

- swarm sizes: `5`, `10`, `15`
- multiple random seeds per condition
- increasing disruption severity per study

## Metrics

Primary metrics:

- recovery time after disruption
- mission completion by deadline
- mission success rate
- recovered abandoned coverage/tasks

Secondary metrics:

- coverage progress over time
- communication overhead
- network fragmentation over time
- role-switch count
- action stability / thrashing rate

## Visualization

Use the same renderer for all methods:

- Main paper figure: 4 grid snapshots from one representative run showing pre-event, disruption, reassignment, and recovered state.
- Supporting plots: coverage-over-time with event markers, recovery-time bars, mission-success bars, and communication-overhead plots.
- Grid visuals: cell color encodes coverage/urgent/blocked/degraded state, and UAV marker color encodes role.
- Optional communication links should appear only in selected snapshots to keep the figure readable.

## LLM Demo Interface

The core experiments remain architecture-first. If a real Qwen or similar LLM demo is added:

- assume one LLM inference completes within one mission tick
- do not claim measured edge-vs-API latency advantages
- use strict JSON I/O only
- validate outputs before execution
- invalid outputs fall back to the safest valid action

Recommended output fields:

- `action`
- `new_role` when applicable
- `target_cell` when applicable
- `messages_to_send`
- `confidence`
- `rationale`

The full prompt, JSON schema, and example I/O should live in the appendix of the paper.

## Verification

- Unit test the method swap contract: changing `method_name` is enough to swap `StaticPartitionMethod`, `RuleAdaptiveMethod`, and `AgenticMethod`.
- Unit test the communication model: messages travel one hop per tick with TTL-limited propagation.
- Unit test event injection: scheduled events mutate world state at the expected tick.
- Unit test rendering: snapshots are generated from simulator state.
- Run all tests with `uv run python -m unittest`.
