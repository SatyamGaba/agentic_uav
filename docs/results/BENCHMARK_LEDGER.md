# Empirical Benchmark Ledger
## Verified Evaluation Results Across Iterations

All metrics in this ledger reflect real simulator runs. No synthesized or mock data is permitted.

> **Dataset correction (2026-11-06):** the earlier ground-VLN-CE entries derived from
> `episode_list_100.json` were invalid (synthetic starts/goals and invalid scene IDs).
> The only valid ground R2R-CE result in this repository is Section 0 below.

---

### 0. Official R2R-CE `val_unseen` Post-Fix Benchmark (2026-11-06)

- **Dataset:** official `R2R_VLNCE_v1-3` `val_unseen`, 17-episode locally available
  stratified subset (`EU6Fwq7SyZv`=7, `oLBMNvg9in8`=7, `pLe4wQe7qrG`=3), seed 42.
- **Source hash:** `d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8`.
- **Harness:** `qwen3.5:4b`, thinking=False, temperature 0.0, 25 simulator actions,
  success radius 3.0 m, duplicate-target + one-shot arrival verification.
- **Evidence:** `audit/phase9_postfix/` (gate PASS for provenance, not performance).

| Metric | Agent | Stop-Immediately Control |
| :--- | :---: | :---: |
| SR | **0.0%** (0/17) | **0.0%** (0/17) |
| OSR | **5.88%** (1/17) | **0.0%** |
| SPL | **0.0** | **0.0** |
| Mean NE | **8.23 m** (95% CI [7.01, 9.47]) | **8.36 m** |
| Mean simulator actions | 8.88 | 0.0 |

Failure taxonomy: `safety_deadlock` 12, `budget_exceeded` 3, `stop_premature` 1,
`protocol_violation` 1.

**Interpretation:** valid negative baseline. The agent did not beat the control.
Many episodes ended with zero actions because the fixed 2D corridor filter rejected
the VLM's above-horizon/wall targets; stair risers are also treated as obstacles.
Do not compare these numbers to the paper's full-split results.

---

### 1. Historical Iteration Baselines (Legacy Aerial / OpenFly; not R2R-CE)

| Run / Framework | Split | Episodes | Threshold | SR (%) | OSR (%) | SPL | Mean NE (m) | Path Length (m) | Compute Backend |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Iteration 3** (Cross-Modal Core) | Seen | 8 | 20.0 m | 37.50% | 50.00% | 0.3641 | 26.82 m | 92.4 m | Local UE4 / AirSim |
| **Iteration 4** (Fast-Slow OmniNav) | Seen | 8 | 20.0 m | 37.50% | 50.00% | 0.3641 | 26.82 m | 88.5 m | Local UE4 / AirSim |
| **Old Baseline** (LMStudio Qwen-VL) | Seen | 1800 | 20.0 m | 16.67% | 27.83% | 0.1294 | 59.08 m | 38.2 m | Remote L40S |
| **Old Baseline** (LMStudio Qwen-VL) | Unseen | 1200 | 20.0 m | 12.67% | 22.42% | 0.0988 | 61.91 m | 34.6 m | Remote L40S |

---

### 2. Refined Framework Verification Runs (Dual-Threshold Reporting)

#### Run ID: `REF-2026-10-26-40EP-SEEN`
- **Date**: 2026-10-26
- **Dataset**: `benchmark/seen.json` (40 Episodes)
- **Environment**: `OpenFlyBenchmarkEnv` (With full 3D visual perspective & depth streaming)
- **Initial Heading**: True `raw_yaws[0]` initialized per episode
- **Action Kinematics**: Canonical OpenFly Discrete Physics (Large=9m, Med=6m, Fine=3m)

| Metric Category | Standard / Threshold | Value | 95% Wilson Confidence Interval |
| :--- | :--- | :--- | :--- |
| **Official OpenFly Benchmark Standard** | **SR @ 20.0 m** | **12.50%** | [5.46%, 26.11%] |
| | **OSR @ 20.0 m** | **22.50%** | [12.32%, 37.50%] |
| | **SPL @ 20.0 m** | **0.1208** | — |
| **Strict Precision Standard** | **SR @ 5.0 m** | **10.00%** | [3.96%, 23.05%] |
| | **OSR @ 5.0 m** | **10.00%** | [3.96%, 23.05%] |
| | **SPL @ 5.0 m** | **0.0958** | — |
| **Flight Dynamics** | **Mean Navigation Error (NE)** | **75.22 m** | — |
| | **Mean Oracle Min Distance** | **47.12 m** | — |
| | **Mean Predicted Path Length** | **88.65 m** | (Ground Truth: 101.92 m) |
| | **Mean Latency per Episode** | **0.09 s** | — |

---

#### Run ID: `REF-2026-10-26-40EP-UNSEEN`
- **Date**: 2026-10-26
- **Dataset**: `benchmark/unseen.json` (40 Episodes)
- **Environment**: `OpenFlyBenchmarkEnv` (With full 3D visual perspective & depth streaming)
- **Initial Heading**: True `raw_yaws[0]` initialized per episode

| Metric Category | Standard / Threshold | Value | 95% Wilson Confidence Interval |
| :--- | :--- | :--- | :--- |
| **Official OpenFly Benchmark Standard** | **SR @ 20.0 m** | **7.50%** | [2.58%, 19.86%] |
| | **OSR @ 20.0 m** | **17.50%** | [8.75%, 31.95%] |
| | **SPL @ 20.0 m** | **0.0722** | — |
| **Strict Precision Standard** | **SR @ 5.0 m** | **5.00%** | [1.38%, 16.50%] |
| | **OSR @ 5.0 m** | **5.00%** | [1.38%, 16.50%] |
| | **SPL @ 5.0 m** | **0.0472** | — |
| **Flight Dynamics** | **Mean Navigation Error (NE)** | **69.96 m** | — |
| | **Mean Oracle Min Distance** | **51.63 m** | — |
| | **Mean Predicted Path Length** | **75.08 m** | (Ground Truth: 105.45 m) |
| | **Mean Latency per Episode** | **0.07 s** | — |

---

### 3. Live Neural VLM Benchmark Runs (Remote NVIDIA L40S - Zero Heuristics)

#### Run ID: `REF-2026-10-26-LIVE-QWEN3VL-L40S`
- **Date**: 2026-10-26
- **Compute Server**: Remote NVIDIA L40S (48GB VRAM), `https://172.24.16.175/user/krutik`
- **Model**: `qwen3-vl:8b` via Ollama (`http://127.0.0.1:11434/api/chat`)
- **Perception**: Closed-loop 3D perspective pinhole camera (RGB frames `360x640` base64) + Float32 metric depth streaming
- **Adapter**: `CalibratedNavAdapter` with multi-subgoal grammatical action-boundary chunking
- **Episodes Completed**: 17 / 40 (Execution paused by institutional 5-hour cluster session quota at 15:40; renewal window opens 15:55)

##### Empirical Episode Telemetry Sample (Episodes 0–16):
| Episode ID | Official 20m Status | Strict 5m Status | Final NE (m) | Oracle Min NE (m) | SPL @ 5m | Steps | Duration (s) | Observations & Landmark Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Ep #000** | FAIL | FAIL | 57.73 m | 57.73 m | 0.0000 | 9 | 133.8 s | Forward corridor traverse |
| **Ep #001** | FAIL | FAIL | 91.82 m | 91.50 m | 0.0000 | 15 | 98.6 s | Multi-segment lateral heading |
| **Ep #002** | **SUCC @ 20m** | **SUCC @ 5m** | **1.61 m** | **1.61 m** | **1.0000** | **15** | **72.7 s** | **Precision landmark arrival (circular garden/red building)** |
| **Ep #003** | FAIL | FAIL | 43.35 m | 38.49 m | 0.0000 | 17 | 71.8 s | Approach overshoot |
| **Ep #004** | FAIL | FAIL | 89.10 m | 89.10 m | 0.0000 | 11 | 150.8 s | Deliberative visual reasoning |
| **Ep #005** | FAIL | FAIL | 56.03 m | 51.26 m | 0.0000 | 18 | 122.8 s | Sector heading adjustment |
| **Ep #006** | FAIL | FAIL | 34.25 m | 34.25 m | 0.0000 | 13 | 122.1 s | Multi-step corridor traverse |
| **Ep #007** | FAIL | FAIL | 81.05 m | 81.05 m | 0.0000 | 22 | 132.9 s | Extended search trajectory |
| **Ep #008** | FAIL | FAIL | 25.67 m | **9.68 m** | 0.0000 | 15 | 110.3 s | **Oracle arrival within 10m threshold** |
| **Ep #009** | **SUCC @ 20m** | FAIL | **6.59 m** | **6.59 m** | 0.0000 | **6** | **53.9 s** | **OpenFly baseline standard arrival** |
| **Ep #010** | FAIL | FAIL | 2058.91 m | 2058.91 m | 0.0000 | 0 | 0.0 s | Token-limit exception safely caught by runner |
| **Ep #011** | **SUCC @ 20m** | **SUCC @ 5m** | **0.00 m** | **0.00 m** | **1.0000** | **2** | **28.8 s** | **Exact waypoint touchdown** |
| **Ep #012** | FAIL | FAIL | 83.69 m | 79.43 m | 0.0000 | 12 | 68.2 s | Heading drift correction |
| **Ep #013** | FAIL | FAIL | 22.16 m | 22.16 m | 0.0000 | 11 | 162.9 s | Approach near boundary |
| **Ep #014** | FAIL | FAIL | 96.89 m | 85.64 m | 0.0000 | 9 | 141.4 s | Visual grounding search |
| **Ep #015** | FAIL | FAIL | 121.58 m | 98.58 m | 0.0000 | 16 | 83.8 s | Lateral drift |
| **Ep #016** | FAIL | FAIL | 84.64 m | 84.64 m | 0.0000 | 9 | 98.8 s | Terminal deceleration leg |

##### Aggregate Partial Metrics (17 Episodes):
- **SR @ 20.0 m**: **17.65%** (3/17) [95% Wilson CI: 6.22%, 41.03%]
- **SR @ 5.0 m**: **11.76%** (2/17) [95% Wilson CI: 3.29%, 34.33%]
- **OSR @ 20.0 m**: **23.53%** (4/17) [95% Wilson CI: 9.57%, 47.26%]
- **Mean Episode Duration**: **97.0 seconds / episode** (Real neural inference on NVIDIA L40S)

### 3. Critical Analytical Findings
1. **Resolution of Trajectory Truncation**: Predicted path lengths increased from $\sim 35\text{m}$ to **$88.65\text{m}$ (seen)** and **$75.08\text{m}$ (unseen)**, successfully covering $87\%$ and $71\%$ of the ground truth distance respectively.
2. **Oracle vs Terminal Divergence**: The gap between Oracle Success ($22.5\%$) and Final Success ($12.5\%$) indicates that the drone frequently passes within the $20\text{m}$ arrival cylinder during flight but continues past before executing terminal stop. Addressing this requires stricter visual arrival confirmation gates.
