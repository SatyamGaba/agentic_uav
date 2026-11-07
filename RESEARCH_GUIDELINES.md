# Operational & Scientific Research Guidelines
## Aerial Vision-and-Language Navigation (VLN) Framework

> **Status**: Active Mandate  
> **Applies to**: All code development, simulation runs, VLM inference, telemetry, documentation, and paper submissions.

> **Ground-replica addendum (2026-11-06):** This file was written for the legacy
> AirSim/OpenFly aerial stack. The active R2R-CE ground replica runs locally in WSL2
> with Habitat-Sim 0.3.3 and Ollama `qwen3.5:4b`; that local execution is authorized.
> R2R-CE success uses the standard 3.0 m radius, not the legacy 20 m/5 m aerial
> thresholds. The zero-mock, zero-synthetic-episode, and metric-integrity rules
> remain mandatory.

---

## 1. Core Operating Principles

### 1.1 Pure Closed-Loop Multi-Modal Execution
1. **Visual Feedback is Mandatory**: Every navigation step must supply real visual observations:
   - High-fidelity RGB scene view.
   - 2D Metric Depth map ($H \times W$ float32 array).
   - Normalized $[u, v]$ screen coordinates or directional visual tokens grounded against visual landmarks.
2. **Dynamic Trajectory Refinement**: Trajectories must not be static pre-compiled paths. The agent must dynamically adjust heading, speed, altitude, and termination based on visual landmark observation and depth verification.
3. **No Blind Runs**: Disabling visual inputs or executing text-only navigation benchmarks under the guise of "VLN" is strictly prohibited.

---

## 2. Infrastructure & Compute Split

### 2.1 Remote Server Mandate for Execution
1. **All VLM Inference on Remote Server**:
   - Model inference (e.g., `qwen3-vl:8b`, `qwen2.5-vl`, `openvla`) must execute on the dedicated remote GPU server (**NVIDIA L40S, 48 GB VRAM**).
   - Local machine (RTX 4050 6GB Laptop) is strictly reserved for:
     - Ideation, code writing, and adapter design.
     - Architecture documentation and statistical post-processing.
     - Fast unit-testing of data structures and contracts.
2. **AirSim UE4 Execution on Remote Server**:
   - Unreal Engine 4 simulation environments (`AirVLN-Linux`) run under headless `xvfb-run` on the remote server with direct GPU RPC integration.

---

## 3. Disaster Recovery & Synchronization Protocol

### 3.1 Pre-Run Synchronization (Local $\to$ Remote)
- Before triggering any remote experiment, push local adapter, environment, and runner updates:
  ```bash
  python sync_framework_to_remote.py
  ```

### 3.2 Post-Run Archival (Remote $\to$ Local)
- Remote JupyterHub containers are routinely subject to reboots, updates, or maintenance sweeps.
- **Rule**: Immediately upon completion of any remote run, synchronize all scripts, logs, dashboards, and JSON summaries back to local disk:
  ```bash
  python sync_from_remote.py
  ```
- Store all historical runs under `remote_backup/` and `benchmark/` with immutable timestamps.

---

## 4. Prohibition of Synthetic Data & Heuristics

### 4.1 Zero Synthesized Data / Fake Benchmarks
1. **No Mock Backends in Evaluation**:
   - Mock VLM backends (`MockVLMBackend`) or canned responses are prohibited for benchmark reports.
   - All published or recorded metrics must stem from live model token generation.
2. **No Heuristic Rule-Based Fallbacks**:
   - Regex-based motion compilations (e.g. mapping "turn right" to hardcoded action lists) are prohibited in adapter evaluation.
   - If a VLM times out, throws an API error, or hallucinates invalid tokens, the episode must log an explicit failure (`sim_failure` or `model_error`). It must **never** silently switch to a heuristic rule engine.
3. **Metric Integrity**:
   - Ground truth starting positions, orientations, and goal coordinates must be extracted faithfully from benchmark splits without synthetic shifting.

---

## 5. Journal-Level Scientific Rigor

### 5.1 Dual-Threshold Evaluation Standard
Every benchmark run must report metrics across **both** operational regimes:
1. **Official OpenFly Benchmark Threshold ($20.0\,\text{m}$)**:
   - For direct, rigorous comparison against published SOTA baselines (OpenFly ICLR 2026, AerialVLN, Iteration 3/4).
   - Metrics: $\text{SR}_{20\text{m}}$, $\text{OSR}_{20\text{m}}$, $\text{SPL}_{20\text{m}}$.
2. **Strict Precision Threshold ($5.0\,\text{m}$)**:
   - Evaluates millimeter/meter-level visual landing or docking precision at terminal landmarks.
   - Metrics: $\text{SR}_{5\text{m}}$, $\text{OSR}_{5\text{m}}$, $\text{SPL}_{5\text{m}}$.

### 5.2 Mandatory Statistical Metrics
All summary tables must compute and display:
- **$\text{SR}$ (Success Rate, %)**: Final position $\le$ arrival threshold.
- **$\text{OSR}$ (Oracle Success Rate, %)**: Any point along trajectory $\le$ arrival threshold.
- **$\text{NE}$ (Navigation Error, meters)**: Mean Euclidean distance from terminal position to goal.
- **$\text{Oracle NE}$ (meters)**: Mean minimum distance achieved along flight.
- **$\text{SPL}$ (Success weighted by Path Length)**: Penalizes inefficient detours.
- **$95\%$ Wilson Score Confidence Intervals**: $[\text{lower}\%, \text{upper}\%]$ for both $\text{SR}$ and $\text{OSR}$.
- **Trajectory Length Dynamics**: Mean predicted path length ($\bar{L}_{\text{pred}}$) vs ground truth ($\bar{L}_{\text{gt}}$) to explicitly detect early-stopping or stalling anomalies.

---

## 6. Living Documentation & Transparency Hierarchy

All progress, anomalies, and findings must be maintained in real-time under structured docs:

| Document / Directory | Mandatory Content |
| :--- | :--- |
| `RESEARCH_GUIDELINES.md` | This document. Living scientific protocol and rules of engagement. |
| `docs/architecture/` | Architectural specifications per iteration (Iter 1 $\to$ Iter 4 $\to$ Refined framework). |
| `docs/results/` | Immutable log of verified experimental runs, model checkpoints, and evaluation summaries. |
| `docs/failures/` | Failure mode taxonomy: categorizes runs by visual grounding drift, altitude loss, collision, or timeout. |
| `docs/plans/` | Explicit, hypothesis-driven plans for upcoming iterations and experiments. |

---

## 7. Pre-Flight Checklist Before Every Benchmark Run

- [ ] Remote GPU connection verified (`check_remote_env.py` returns L40S active, Ollama running).
- [ ] VLM Vision backend verified (`qwen3-vl:8b` responding with valid multi-modal chat).
- [ ] AirSim / UE4 simulation active with valid RGB and Depth buffer streaming.
- [ ] Start pose and start heading correctly initialized from `raw_yaws[0]`.
- [ ] Mock backend flag disabled; heuristic adapter disabled.
- [ ] Post-run auto-sync scheduled to pull remote results to local machine.
