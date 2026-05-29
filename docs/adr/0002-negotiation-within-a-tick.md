# Negotiation resolves within a single tick

## Status

accepted

## Decision

When an event fires (`dropout`, `urgent_sector`, `block_sector`), the affected
UAVs negotiate who-does-what — propose / counter / settle — **synchronously within
that one decision tick**. The negotiation completes before the tick's actions are
applied. Conflicts resolve deterministically: the **nearest UAV wins**, ties
broken by lexicographic `uav_id`. The one-hop-delayed `NetworkModel` continues to
govern ordinary mission-execution messaging *between* events.

## Context

A real disaster swarm cannot reach a multi-round agreement instantaneously, so
modeling negotiation as zero-cost would be a cheat. But forcing every negotiation
to play out over many one-hop-delayed message rounds would dominate the
simulation and entangle the comparison with networking artifacts rather than
decision quality. We needed a single, defensible modeling assumption.

The resolution: a **tick maps to a coarse real-world window** — larger cells,
longer wall-clock intervals between decisions (see CONTEXT.md `tick`). At that
granularity, one negotiation round genuinely fits inside a tick. So in-tick
synchronous negotiation is a faithful abstraction, not a shortcut.

## Considered options

- **Multi-tick negotiation over the delayed network.** Rejected: makes the
  comparison about message latency, not reasoning, and bloats every event into a
  multi-tick stall.
- **Zero-cost / instantaneous global negotiation everywhere.** Rejected: implies
  free global coordination, contradicting the decentralized, range-limited model
  in ADR-0001.
- **Synchronous in-tick negotiation only on events** (chosen): bounded, cheap,
  and justified by the coarse-tick assumption; ordinary messaging stays delayed.

## Consequences

- Between events, messaging is still subject to the `NetworkModel`'s one-hop
  delay and communication range — in-tick resolution is reserved for
  event-driven negotiation.
- Deconfliction is deterministic (nearest wins, tie by `uav_id`), reusing the
  existing `_choose_unclaimed` / `peer_id < uav_id` convention, so results are
  stable across runs given the same world.
- The validity of the comparison rests on the coarse-tick assumption; if ticks
  were reinterpreted as fine-grained real-time steps, this ADR would need to be
  revisited.
