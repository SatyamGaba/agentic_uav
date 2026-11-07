# AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness

A zero-mock, end-to-end local replication of **AgenticNav (arXiv:2606.10577, 2026)** for
ground VLN-CE, evaluated in Habitat-Sim 0.3.3 on Matterport3D meshes with local
`qwen3.5:4b` weights.

This repository now reports a **negative but honest baseline**: the harness is
instrumented and provenance-clean, but the current 4B model + fixed standoff safety
filter succeeds on **0/17** official R2R-CE `val_unseen` episodes. See the
[post-fix result](#5-post-fix-r2r-ce-benchmark-result).

---

## 1. Dataset Correction (2026-11-06)

The earlier `agenticnav_paper_replica/data/episode_list_100.json` is **not R2R-CE** and
must not be used as benchmark evidence:

- every episode starts at `[0, 0, 0]`;
- goals are synthesized along `-Z`;
- instructions are templated;
- 5/11 scene IDs are invalid.

That file is retained only as deprecated history and is marked
`"deprecated": true, "not_r2rce": true`. All pre-fix `audit/phase9/` results are
therefore scientifically invalid.

The official source is **R2R-CE `R2R_VLNCE_v1-3`, `val_unseen`**:

- 1,839 episodes / 11 scenes;
- source SHA-256 `d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8`;
- locally available scenes: `EU6Fwq7SyZv`, `oLBMNvg9in8`, `pLe4wQe7qrG`;
- 327 locally usable episodes.

The sealed benchmark subset is
`agenticnav_paper_replica/data/r2rce_val_unseen_17.json` (17 episodes, stratified
EU6=7 / oLB=7 / pLe=3, seed 42), built reproducibly by
`scripts/build_r2rce_subset.py`.

---

## 2. Architecture

- **Perception:** four panoramic RGB-D views (640x480, 90 deg HFOV) + BEV occupancy map.
- **Tools:** `query_depth`, `recall`, `move_to`, `stop`.
- **Safety:** 2D swept-corridor collision check with a 0.30 m fixed standoff and
  0.25 m forward quantization (spec: `spec/spec_ledger.yaml`).
- **Privilege firewall:** prompts receive only sensory data. Goal coordinates and
  NavMesh are used only for post-episode evaluation.
- **Backends:** Habitat-Sim 0.3.3 (real scenes only), Ollama `qwen3.5:4b`
  (vision enabled, `thinking=False`, temperature 0.0, seed 42).
- **No fallbacks:** no mock VLM, no synthetic episodes, no scene substitution.

---

## 3. Repository Layout

```text
UAV/
├── agenticnav_paper_replica/      # Ground VLN-CE implementation
│   ├── configs/                   # ReplicaConfig
│   ├── data/                      # Official 17-episode subset + deprecated list
│   ├── embodiment/                # HabitatGroundEmbodiment
│   ├── evaluation/                # SR/SPL/OSR/NE/TL + failure taxonomy
│   ├── geometry/                  # Pinhole, standoff, corridor safety
│   ├── harness/                   # Step loop, prompts, tools, executor
│   ├── mapping/                   # Occupancy grid + BEV renderer
│   ├── memory/                    # Image cache + action history
│   ├── traces/                    # JSONL trace logger
│   └── vlm/                       # OllamaNavClient
├── audit/                         # Evidence; phase9 pre-fix is invalid
│   └── phase9_postfix/            # Valid post-fix evidence
├── scripts/
│   ├── build_r2rce_subset.py      # Reproducible official subset builder
│   └── benchmark_r2rce_postfix.py # Sealed benchmark + control baseline
├── spec/spec_ledger.yaml
└── archive/legacy_airsim_uav/     # Legacy UAV code (dependencies ignored)
```

---

## 4. How to Run

```bash
export PYTHONPATH=$(pwd)
pytest agenticnav_paper_replica/ -q

# 17-episode official subset + stop-immediately baseline
python scripts/benchmark_r2rce_postfix.py

# Main run only, or baseline only
python scripts/benchmark_r2rce_postfix.py --no-baseline
python scripts/benchmark_r2rce_postfix.py --baseline-only
```

The benchmark writes `manifest.json`, per-episode JSON, JSONL traces, a control
baseline, and `phase9_postfix_gate_summary.json` to `audit/phase9_postfix/`.
The gate checks provenance/completeness, **not** performance.

---

## 5. Post-Fix R2R-CE Benchmark Result

Run date 2026-11-06, `qwen3.5:4b`, 17 official `val_unseen` episodes, 25 simulator
actions, success radius 3.0 m.

| Metric | Value |
| :--- | :--- |
| Success Rate (SR) | **0.0%** (0/17), Wilson 95% CI [0.0, 18.4] |
| Oracle Success Rate (OSR) | **5.88%** (1/17), Wilson 95% CI [1.05, 26.98] |
| SPL | **0.0** |
| Mean Navigation Error | **8.23 m** (bootstrap 95% CI [7.01, 9.47]) |
| Mean simulator actions | 8.88 |
| Stop-Immediately baseline SR | **0.0%** |
| Stop-Immediately baseline NE | **8.36 m** |

Failure taxonomy: `safety_deadlock` 12, `budget_exceeded` 3, `stop_premature` 1,
`protocol_violation` 1.

The agent was not better than the stop-immediately control. Many episodes ended
with zero simulator actions because the VLM selected above-horizon/wall points and
the fixed 2D corridor filter rejected every candidate. The same filter treats stair
risers as obstacles, so stair-heavy routes are especially brittle. This is a
behavioral/embodiment limitation of the current ground replica, not a metric or
provenance failure.

---

## 6. Status and Limitations

- **Provenance gate:** PASS. 17/17 episodes completed, 17 traces, valid taxonomy,
  every deadlock has an arrival-verification turn.
- **Scientific performance:** negative. Do not claim parity with the paper.
- **Dataset:** 17-episode local subset, not the full 1,839-episode split.
- **Renderer:** actual runs used Mesa `llvmpipe` software OpenGL in WSL, not the
  RTX hardware path claimed in older documents.
- **Pre-fix evidence:** `audit/phase9/` is retained untouched but is invalid.
