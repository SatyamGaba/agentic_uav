# Agentic UAV Swarm Simulator

A tick-based 2D disaster-mapping simulator that runs several decentralized swarm
decision methods over an identical world so they can be compared as a fair
experiment harness. This glossary fixes the domain language so that "agentic",
"greedy", "sector", "intent", and the information boundary all mean exactly one
thing across the code, the docs, and the paper.

## Language

### Methods

**agentic**:
The PROPOSED decision method: an LLM-driven, decentralized agent that each UAV
runs from its own local view. It is the paper's contribution; the thesis is that
it beats the heuristic baselines on recovery after disruption.
_Avoid_: "the heuristic", "the smart heuristic", "the planner". Never use
"agentic" for the old greedy method (see **greedy**).

**greedy**:
A heuristic baseline (`GreedyMethod`) that sends each UAV to the nearest open
urgent **sector**, else the nearest uncovered one. This is the renamed former
`AgenticMethod` — it was misnamed; it is NOT the contribution.
_Avoid_: calling it "agentic", "the proposed method", or "the LLM method".

**baseline**:
The four heuristic methods: **static**, **rules**, **task_consideration**, and
**greedy**. The bar that **agentic** must clear.
_Avoid_: "control", "reference method" used loosely to include agentic.

**static**:
A baseline that pre-partitions the grid into per-UAV serpentine **assignments**
at mission start and visits them in order. The partition is over **individual
sectors (cells)** handed out as an ordered per-UAV list — not a carve-up into
contiguous regions, and no region is swept.
_Avoid_: "fixed partition", "static planner", "region assignment".

**rules**:
A baseline (`RuleAdaptiveMethod`) that reacts with hand-written rules: chase
known urgent sectors, else cover locally, else patrol.
_Avoid_: "reactive method", "rule engine".

**task_consideration**:
A baseline (`TaskConsiderationMethod`) that scores candidate **targets** and
deconflicts via a broadcast **commitment** (`task_commitment`).
_Avoid_: "auction method", "bidding".

### World

**sector**:
A single grid cell `(x, y)` — the atomic unit of the world. Every position in
the grid is exactly one sector. A sector is NOT a multi-cell region (see the
"sector = cell, not a region" flag below).
_Avoid_: using "task" or "target" to mean the cell itself; a sector only becomes
a target once a UAV commits to it (see **target**); using "sector" for a region.

**target** (also **task**):
A **sector** that a specific UAV has chosen to cover or respond to. "Target" and
"task" are the same thing: a sector with a UAV's intention attached.
_Avoid_: "destination", "goal cell", or conflating it with the bare sector.

**coverage**:
Per-sector progress toward being fully mapped, in `[0.0, 1.0]`; a sector is
covered at `>= 1.0`. Coverage is **monotonic** — it only ever increases.
_Avoid_: "explored", "scanned" used to mean something different from coverage.

**priority** (sector priority `normal | urgent`):
A sector flagged **urgent** demands a priority responder. The set of urgent
sectors changes only via injected **events**, not via UAV action.
_Avoid_: "hot cell", "high-value" — say **urgent sector**.

**blocked**:
A sector a UAV may not enter (obstacle / no-fly). Part of the static layout
unless changed by a `block_sector` **event**.
_Avoid_: "wall", "obstacle cell" — say **blocked sector**.

### Roles

**role** (`coverage | priority_responder | relay`):
The job a UAV is currently doing. **coverage** = mapping uncovered sectors;
**priority_responder** = servicing an urgent sector; **relay** = positioning to
carry messages between otherwise-disconnected peers.
_Avoid_: "mode", "state" — say **role**. Roles are exactly these three values.

### Communication and decision-making

**intent** (`intent_summary` message):
A broadcast, **non-binding** statement of a UAV's current **target** and **role**
("I'm heading to (3,4) as priority_responder"). Peers use it as a hint to avoid
piling onto the same target; nobody is obligated by it.
_Avoid_: treating intent as a claim or reservation. It is advisory only.

**commitment** (`task_commitment` message):
**task_consideration**'s **binding-ish, scored** claim on a **target** — it
carries a score so peers can deterministically yield to the better-placed UAV.
This is stronger than an **intent**: it is a contested claim with a tiebreak,
not a heads-up.
_Avoid_: using "commitment" and "intent" interchangeably. Intent = non-binding
hint; commitment = scored claim used to resolve conflicts.

**negotiation**:
The process by which UAVs affected by an **event** settle who-does-what
(propose / counter / settle), resolved **synchronously within one tick**.
Conflicts resolve deterministically: nearest UAV wins, ties broken by `uav_id`.
_Avoid_: "auction", "consensus protocol" — and do not imply it spans multiple
ticks (see ADR-0002).

**message** / **NetworkModel**:
A `Message` is a typed payload (`urgent_sector`, `intent_summary`,
`task_commitment`, ...) delivered by the `NetworkModel` with a **one-hop delay**
to peers inside communication range. Governs ordinary mission-execution
messaging between events.
_Avoid_: assuming instant or global delivery.

**event** (`CommunicationEvent`) / **event-triggered re-plan**:
A scheduled disruption injected into the world: `dropout` (UAV lost),
`urgent_sector` (a sector becomes urgent), `block_sector` (a sector becomes
impassable). The **agentic** method re-plans only at mission start and when an
event fires (event-triggered), not every tick.
_Avoid_: "step", "incident" — say **event**. Do not say agentic runs "per tick".

**tick**:
One discrete decision/simulation step. Modeled as a **coarse real-world window**
(large cells, long intervals) — long enough that a full **negotiation** round
fits inside one tick.
_Avoid_: equating a tick with a wall-clock second or one UAV movement only.

### Information model

**observation**:
The per-UAV input to a decision. Every method receives the SAME observation for
a given UAV — methods differ only in how they reason over it, never in what they
can see.
_Avoid_: giving any method extra fields. Symmetry is non-negotiable (ADR-0001).

**observation symmetry**:
The invariant that all methods (baselines and agentic) get identical observation
keys and payloads. The previous code violated this with a global urgent list and
full-map scans; that is fixed.
_Avoid_: "fairness flag", ad-hoc per-method observation tweaks.

**information boundary**:
The split between what is global and what must be discovered. The **static grid
layout** (dimensions, which sectors are blocked, in-bounds checks) is **global**
— UAVs "have a map". **Dynamic state** (coverage progress + which sectors are
urgent) is NOT global: it must be **discovered** by local sensing or **learned**
via range-limited peer **messages**.
_Avoid_: letting any decision read global dynamic state. Layout = global;
coverage/urgent = local.

**belief map** (`BeliefMap`):
A single UAV's private, accumulated memory of the dynamic state it has personally
sensed or heard about (`covered`, `known_urgent`, `cleared_urgent`). Decisions
read the belief map, never the ground-truth world. Belief lives on
**MethodState**, not on `UavState`, so the world and GUI stay method-agnostic.
_Avoid_: "knowledge", "world model" — say **belief map**. It is per-UAV and
possibly stale/incomplete.

**WorldState** / **ground truth**:
The authoritative dynamic state of the world, written centrally for sensing and
metrics. It is the source UAVs sense FROM, but no decision path reads it
directly.
_Avoid_: reading `world.sectors[*].coverage`/priority inside a decision — that is
the unfair oracle ADR-0001 removes.

## Flagged ambiguities

- **sector vs target/task**: A **sector** is the cell. It becomes a **target**
  (= **task**) only when a UAV commits to covering/responding to it. Do not use
  "task" for an arbitrary cell.
- **intent vs commitment**: **intent** (`intent_summary`) is a non-binding
  broadcast hint; **commitment** (`task_commitment`) is a scored, contested claim
  used for deterministic conflict resolution. They are different message types
  and different strengths.
- **agentic vs greedy**: **agentic** is the LLM proposed method; **greedy** is the
  renamed former heuristic, now a baseline. Historically the greedy heuristic was
  called "agentic" — that usage is retired.
- **global vs local**: only the static **layout** is global. **Coverage** and
  **urgent** are local, reached through **belief map** + **messages**.
- **sector = cell, not a region**: There are exactly TWO spatial tiers — the
  **sector** (a single `(x, y)` cell, the atomic unit) and the whole map
  (**WorldState** / the "area"). There is NO middle "region of cells" tier:
  nothing is decomposed into multi-cell zones, no UAV owns a region, and no
  method sweeps a region's interior. Movement is a one-cell-per-tick greedy
  Manhattan step (`_move_toward`), not A* path-planning, and a sector is
  "covered" the instant it is sensed (no interior to sweep). If you mean the
  atomic square, say **sector** (or **cell**); if you mean the full grid, say
  **the area** / the **world**. Do not use "sector" for a contiguous region of
  cells.

## Example dialogue

**Dev:** When a sector turns urgent, every UAV should immediately re-route to it,
right?

**Expert:** No — that would leak global state. A sector turning urgent is an
**event**, and the world's **ground truth** records it. But a UAV only *knows*
the sector is urgent if it senses that sector locally or hears an
`urgent_sector` **message** from a peer in range. Until then it isn't in that
UAV's **belief map**, so the UAV can't target it.

**Dev:** So the urgent sector is the **target**?

**Expert:** It's a **sector** that *becomes* a **target** once a UAV commits to
responding to it as `priority_responder`. Before anyone commits, it's just an
urgent sector on whoever has sensed it. Keep "sector" for the cell and "target"
for the cell-plus-intention.

**Dev:** And when two UAVs both want it, the `intent_summary` they broadcast
stops the collision?

**Expert:** The **intent** is only a hint — non-binding. It nudges peers away,
but it doesn't reserve anything. If you need a real claim, that's
**task_consideration**'s **commitment** (`task_commitment`), which carries a
score, so the worse-placed UAV deterministically yields. For the **agentic**
method, the actual deconfliction happens through **negotiation**: the affected
UAVs propose and settle **within the same tick**, nearest wins, ties by
`uav_id`.

**Dev:** Why can negotiation finish in one tick but messages have a one-hop
delay?

**Expert:** Because a **tick** is a coarse real-world window — big cells, long
intervals — so one negotiation round fits inside it. Between events, ordinary
mission messages still ride the `NetworkModel` with its one-hop delay. The
agentic LLM only re-plans at mission start and on **events**; the rest of the
time UAVs just execute the persisted plan.

**Dev:** And all of this is the same for the greedy baseline?

**Expert:** Same **observation**, yes — that's **observation symmetry**, and it's
non-negotiable. **greedy** (the old heuristic we stopped calling "agentic") sees
exactly what **agentic** sees. The only difference is the reasoning on top.
