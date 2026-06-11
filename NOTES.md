# Teaching Notes

## About the learner
- Author of most of the repo (git history shows their commits on branch `sim`).
- Goal is **thesis defense**, not onboarding. Pitch above file-by-file description;
  emphasize *why* design choices exist and how they protect the central claim.
- Asked for the high-level structure first ("whole-system map").

## Teaching preferences
- One tightly-scoped win per lesson; keep within working memory.
- Ground every claim in a repo artifact (file:line, ADR, spec) — they can check it.
- No code changes during teaching sessions (explicit instruction, 2026-06-11).

## Repo facts worth remembering (verified 2026-06-11)
- Modules ON DISK in `agentic_uav/`: agent_core, communication, environment,
  experiments, gui, gui_support, llm_agent, llm_providers, methods, models,
  planning, policy, rendering, safety, scenarios, simulation, tools, types,
  ui_config, __init__.
- `SwarmMethod` interface + all 5 methods + `build_method()` live in `policy.py`
  (NOT separate files). README/CLAUDE.md sometimes imply otherwise.
- The LLM seam is split: `llm_agent.py` (AgentDecider protocol, MockAgentDecider,
  serialize/validate, build_decider) and `llm_providers.py` (real Bedrock/QGenie
  adapters, opt-in via the `llm` extra / boto3).
- **Doc tension to handle honestly:** `report.md` is an earlier audit that says
  `handle_event()` is never called and `intent_summary` is emit-only. CLAUDE.md,
  ADR-0001/0002/0003, `docs/architecture.mmd`, and `docs/NEXT_STEPS.md` all mark
  those as RESOLVED. New files agent_core/safety/tools/environment map onto the
  gaps the audit flagged. Treat report.md as "snapshot, then" vs. "now".
- Canonical diagram: `docs/architecture.mmd`. ADRs: `docs/adr/000{1,2,3}-*.md`.
- 27 related-work PDFs in `related-work/pdfs/` (Chen 2023 is the cited basis for
  the `task_consideration` baseline).

## Lesson ideas (backlog)
- L2: The 5-method seam + the swappability invariant (only method_name changes).
- L3: The two invariants up close — observation symmetry & decentralization (the
  fairness ADR), framed as "what a reviewer attacks."
- L4: The agentic LLM seam — decider protocol, mock default, validate-or-fallback,
  event-triggered replan + in-tick negotiation.
- L5: Spec vs. implementation — the audit, what's since been wired, what's open
  (the honest "state of the thesis" lesson).
