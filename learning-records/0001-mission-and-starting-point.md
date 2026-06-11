# Mission established: defend the thesis; learner is the repo author

The learner authored most of `agentic_uav` and wants to learn its high-level structure
**to defend the research thesis** (LLM `agentic` method beats heuristic baselines on
recovery after disruption) to advisors/reviewers. This is articulation/consolidation,
not cold onboarding — pitch lessons above "what does this file do" and emphasize *why*
design choices protect the central claim.

First lesson delivered: `0001-whole-system-map.html` — the seven-beat tick loop, the five
layers and their modules, and the two load-bearing invariants (observation symmetry,
decentralization), framed as "what a reviewer attacks."

## Implications
- Next sessions should drill into single boxes on the map (the SwarmMethod seam; the two
  invariants up close; the LLM seam; spec-vs-implementation honesty).
- Handle the `report.md` vs. ADR/CLAUDE.md tension explicitly: the audit's "handle_event
  never called / intent_summary emit-only" gaps are marked resolved in newer docs. The
  honest defense currently rests on *architecture & fairness*, not a measured win (the
  default decider is a mock that mirrors `greedy`).
- No code changes during teaching (explicit instruction). See [[MISSION.md]].
