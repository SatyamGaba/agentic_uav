# Scope-up: full 7-lesson navigable website for phone reading

User asked (2026-06-11) to build out ALL lessons as a navigable website readable on the phone,
using subagents, and to add a final lesson on a centralized global-info LLM decision layer.

Delivered structure (workspace `lessons/`):
- `lesson.css` — shared, mobile-first/print-friendly stylesheet. L4–L7 + homepage link it;
  L1–L3 keep equivalent inline styles (not refactored, to avoid risk — they render identically).
- `index.html` — mobile homepage / table of contents; the nav hub. Cards link all 7 lessons +
  reference/primary-source links (glossary, architecture.mmd, ADRs, EXPERIMENTS.md, MISSION.md).
- Every lesson has top + bottom `.lessonnav` (Prev / Contents / Next). Phone CSS hides the middle
  "Contents" link under 520px so Prev/Next stay thumb-reachable.

Curriculum (all tie to thesis defense):
L1 whole-system map · L2 SwarmMethod seam · L3 invariants + oracle rebuttal ·
L4 the LLM seam · L5 measuring recovery · L6 state of the thesis (honest ledger) ·
L7 centralized global-info LLM layer (PROPOSED contrast/upper bound — NOT in code).

## Key framing decision (L7)
The centralized global-info LLM is presented as a deliberate upper-bound BASELINE / future work,
explicitly NOT implemented. It is literally the global oracle ADR-0001 removed — legitimate when
reported as a separate, labelled condition, illegitimate only when hidden inside every method.
Its value: gives a ceiling the decentralized agent is measured against, and pre-empts "why not
centralize?" with the resilience trade-off (single point of failure, comms dependence,
scalability) under dropout/comm-blackout.

## Method facts locked for L4/L5 (verified via subagents, line-referenced)
- LLM seam: AgentDecider protocol (llm_agent.py:26), build_decider default "mock" (:32-60),
  pipeline serialize→call→validate→fallback (decide_proposal :124; validate_action :105),
  MockAgentDecider mirrors greedy (:179-232, docstring :180-184; test :153). Providers Bedrock/
  QGenie in llm_providers.py (lazy, opt-in). Honest caveat: agentic ≈ greedy today, no measured win.
- Recovery: recovery_slope (experiments.py:306-330), recovery_ticks_to_target (:333-353);
  coverage is MONOTONIC so recovery = continued progress under reduced capacity (EXPERIMENTS.md:19-26);
  paired seeded sweep, mean+SEM only (no significance tests — a gap).

See [[0002-swarmmethod-seam-delivered]] and [[0001-mission-and-starting-point]].
