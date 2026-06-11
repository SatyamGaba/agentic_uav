# agentic_uav Framework Resources

These are the highest-trust sources *inside this repo* for understanding the
framework's structure and defending its thesis. Prefer them over outside summaries —
they are the project's own canonical record.

## Knowledge (in-repo, canonical)

- [Design spec — `docs/superpowers/specs/2026-04-26-swappable-uav-swarm-simulator-design.md`](docs/superpowers/specs/2026-04-26-swappable-uav-swarm-simulator-design.md)
  The authoritative design. Use for: the SwarmMethod contract, the Decentralization
  Principle, the per-UAV agent architecture (sense→think→act), and experiment design.
- [ADR-0001 — Decentralization & observation symmetry](docs/adr/0001-decentralization-and-observation-symmetry.md)
  The single most important document for thesis defense. Use for: why a global urgent
  list was removed, why belief lives on `MethodState` not `UavState`, and the exact
  argument that kills "the agentic win is just an oracle."
- [ADR-0002 — Negotiation within a tick](docs/adr/0002-negotiation-within-a-tick.md)
  Use for: how colliding proposals deconflict synchronously via `intent_summary`.
- [ADR-0003 — Event-triggered cadence & provider-agnostic adapter](docs/adr/0003-event-triggered-cadence-and-provider-agnostic-adapter.md)
  Use for: why `agentic` re-plans on events (not every tick) and why the LLM sits
  behind a swappable adapter with a mock default.
- [`docs/architecture.mmd`](docs/architecture.mmd)
  The canonical end-to-end flowchart (Mermaid). Use for: the authoritative module wiring
  diagram. Renders at https://mermaid.live if you paste it in.
- [`README.md`](README.md) — the thesis statement and the 5-method summary, verbatim.
- [`CLAUDE.md`](CLAUDE.md) — current invariants and which review gaps are now closed.
- [`docs/NEXT_STEPS.md`](docs/NEXT_STEPS.md) — roadmap; the 2026-05-28 status block lists
  what is DONE vs. open.
- [`report.md`](report.md) — an earlier spec-vs-implementation audit. **Read critically:**
  some "gaps" (handle_event never called, intent_summary emit-only) are marked resolved
  elsewhere. Use for: the honest list of what still needs work — but cross-check against
  the ADRs and CLAUDE.md, which are newer.

## Knowledge (external, the baselines' basis)

- [`related-work/pdfs/01-chen2023decentralized.pdf`](related-work/pdfs/01-chen2023decentralized.pdf)
  Chen, Li & Peng (2023). The `task_consideration` baseline (Baseline C) is inspired by
  this. Use for: defending why that baseline is a credible, strong opponent to beat.
- `related-work/pdfs/` (27 PDFs total) — LLM-agent foundations (ReAct 15, Reflexion 16,
  Voyager 17, Toolformer 19) and multi-agent coordination (RoCo 24, SMART 25, LAMMA-P 28).
  Use for: situating the `agentic` method in the literature when a reviewer asks "what's new."

## Wisdom (Communities)

*Not yet surfaced.* For thesis defense, the relevant "community" is your advisor/committee
and the venue's reviewers — the real test bed. If you want online corroboration for the
multi-agent-LLM framing, candidate venues to watch are the multi-robot-systems and
LLM-agents tracks (e.g. CoRL, AAMAS, NeurIPS agent workshops). Tell me if you'd like me
to find specific high-signal forums or reading groups and I'll add them here.

## Gaps
- No live experimental result yet supporting the thesis: the default decider is a
  deterministic **mock** that mirrors `greedy`, so a real `agentic`-vs-baselines recovery
  number requires wiring a vendor behind the adapter (see NEXT_STEPS item 4). Defense
  must currently rest on *architecture and fairness*, not on a measured win.
