# L2 delivered: the SwarmMethod seam and swappability

Delivered `0002-swarmmethod-seam.html`: the three-member `SwarmMethod` Protocol, the single
swap point (`build_method`, policy.py:468), proof the harness is method-blind (grep shows
`method_name` appears once in simulation.py:149; the method is only touched via the three
contract calls at simulation.py:143/177/197), a six-step observation→Action trace, and the
"same MethodState type, different fields" table.

Reframed the thesis defense around swappability-as-evidence: a method-blind harness cannot be
rigged toward `agentic`.

## Non-obvious thing established (worth carrying forward)
The `report.md` "agentic is the simplest method (~15 lines)" finding is now STALE. In current
`policy.py`, `agentic` is the LARGEST method (~180 lines, policy.py:211–390) — the only one
with a replan/execute split, in-tick deconfliction, and plan persistence. BUT the lesson is
honest that this is orchestration, not smarter per-UAV reasoning: the choice is delegated to
the `decider`, and the default `MockAgentDecider` still mirrors `greedy`. Defense rests on
architecture, not a measured win. This complexity-inversion point is a ready rebuttal to the
old audit. See [[0001-mission-and-starting-point]].

## Spacing/interleaving applied
L2's quiz Q1 re-tests L1's "only beat 4 differs" to build storage strength via spacing.

## Next
L3 — the two invariants up close (ADR-0001): locate where global `urgent_cells` was removed;
rehearse the oracle rebuttal as a defense drill.
