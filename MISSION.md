# Mission: The agentic_uav framework's high-level structure

## Why
We're defending this project's research thesis — that an **LLM-driven, decentralized
`agentic` method beats heuristic baselines on recovery after disruption** — and we want
the argument airtight before it meets advisors and reviewers. That means being able to
explain, on demand, exactly how the framework is built, *why* each design choice protects
the fairness of the claim, and where the current implementation does and does not yet
support it. This is shared prep between co-authors, not coaching.

## Success looks like
- We can draw the whole-system data flow from memory (config → tick loop → per-UAV
  decision → resolve → sense → metrics) and name the module each piece lives in.
- We can state the two load-bearing invariants — **observation symmetry** and
  **decentralization** — and explain why any challenge to the thesis is really a
  challenge to one of them.
- We have a ready answer to "isn't the agentic advantage just a global oracle?" — the
  specific mechanism that rules it out (per-UAV `BeliefMap`, `urgent_cells` removed).
- We can honestly characterize the implementation's current state vs. the spec — what
  the audit flagged, what has since been wired, and what is still open.

## Constraints
- The user authored most of this code; this is consolidation/articulation, not cold
  onboarding. Lessons should aim above "what does this file do."
- No code changes as part of teaching — lessons produce docs/HTML only.
- Working memory is small: one tightly-scoped win per lesson.

## Out of scope (for now)
- Low-level UAV aerodynamics / flight control (the sim is mission-level by design).
- Writing the paper's prose, plots, or statistics.
- Standing up a real LLM vendor behind the adapter (a known open item, not today's lesson).
