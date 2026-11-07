# AgenticNav Local Replication: End-to-End Architectural Verification & Audit Report

**Date of Execution:** 2026-11-06  
**Host Platform:** Windows 11 + WSL2 (Ubuntu 24.04 LTS)  
**Simulator Backend:** Habitat-Sim 0.3.3; post-fix runs used Mesa `llvmpipe` software OpenGL  
**Embodied Dataset:** official R2R-CE `R2R_VLNCE_v1-3` `val_unseen` subset (`EU6Fwq7SyZv`, `oLBMNvg9in8`, `pLe4wQe7qrG`)  
**Vision-Language Model:** local Ollama `qwen3.5:4b`, `thinking=False`, temperature 0.0, seed 42  

> **POST-FIX ADDENDUM (2026-11-06):** Sections 1-5 below describe the **pre-fix**
> synthetic episode list and are retained only as history. That list was not R2R-CE
> (all starts `[0,0,0]`, synthesized goals, 5/11 invalid scene IDs), so all pre-fix
> `audit/phase9/` numbers are **invalid**. The valid post-fix result is in
> Section 0 and `audit/phase9_postfix/`.

---

## 0. Post-Fix Dataset Correction and Official Benchmark

### 0.1 Dataset corrected
- Official source: `R2R_VLNCE_v1-3/val_unseen` (1,839 episodes, 11 scenes), SHA-256
  `d173d8028537f30ab652dc5d24ead737e2b6010b6a4599f974351685710d18e8`.
- Locally available scenes: `EU6Fwq7SyZv`, `oLBMNvg9in8`, `pLe4wQe7qrG`; 327 usable
  episodes.
- Sealed subset: `agenticnav_paper_replica/data/r2rce_val_unseen_17.json`, 17
  episodes (EU6=7, oLB=7, pLe=3), stratified seed 42, built by
  `scripts/build_r2rce_subset.py`.
- `episode_list_100.json` is marked deprecated and is not benchmark evidence.

### 0.2 Post-fix run (valid)
- Harness: fixed missing executor import, fail-fast VLM backend, canonical failure
  taxonomy, duplicate-target detection, one-shot arrival verification, no repeat
  target prompt, `qwen3.5:4b`, 25 simulator actions, success radius 3.0 m.
- Gate: **PASS** for provenance/completeness (`audit/phase9_postfix/phase9_postfix_gate_summary.json`).
- Result: **SR 0.0% (0/17)**, **OSR 5.88% (1/17)**, **SPL 0.0**, mean NE **8.23 m**
  (bootstrap 95% CI [7.01, 9.47]), mean 8.88 simulator actions.
- Failure taxonomy: `safety_deadlock` 12, `budget_exceeded` 3, `stop_premature` 1,
  `protocol_violation` 1.
- Control: stop-immediately baseline **SR 0.0%**, mean NE **8.36 m**.
- Interpretation: the agent was not better than the control. Many episodes ended
  with zero simulator actions because the VLM selected above-horizon/wall points and
  the fixed 2D corridor filter rejected every candidate; stair risers are also
  treated as obstacles. This is a behavioral/embodiment limitation, not a metric or
  provenance failure.

### 0.3 Pre-fix evidence status
- `audit/phase9/` is preserved but **invalid** as R2R-CE evidence. Do not cite its
  SR/OSR/NE values.
- Its claimed `100% OSR` / `2.33 m NE` came from the synthetic list and are not
  comparable to the official split.

---

## 1. Executive Summary (PRE-FIX, SUPERSEDED)

This report documents the rigorous local verification, baseline audit, and failure root-cause analysis of **AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness (arXiv:2606.10577, 2026)** conducted on local hardware.

All mock fallbacks were eliminated. All perception, rendering, and geometry operations execute directly within real Habitat-Sim 0.3.3 continuous environments on real Matterport3D photorealistic meshes.

### Core Discoveries:
1. **Root Cause of SR 0% vs OSR 100% Uncovered and Resolved:**
   - In the initial 15-episode benchmark, the agent consistently reached the goal area in every episode ($\text{OSR} = 100.0\%$, mean navigation error $2.33\text{ m}$, reaching as close as $0.18\text{ m}$ / 18 cm from the target), but registered $\text{SR} = 0.0\%$.
   - Root-cause analysis traced this to a two-fold trap:
     a) **The Standoff & Quantization Deadlock Trap:** At close range to terminal furniture/walls ($D_{\text{raw}} \approx 0.50\text{ m}$), standoff subtraction ($0.30\text{ m}$) leaves $\Delta d = 0.20\text{ m} < 0.25\text{ m}$ (forward step resolution), resulting in $\rho_q = 0.0\text{ m}$ (`no_motion`). The agent retried forward movement 5 times consecutively, exhausting $R_{max}=5$ and aborting without calling `stop`.
     b) **The Prompt Prohibitions:** The initial prompt contained an artificial anti-premature-stop rule (`accumulated_distance_m < 6.0m`), commanding the model to *never* stop under 6m. Because the mean evaluation route length was only $2.70\text{ m}$ (with 10 of 15 episodes starting $\le 3.0\text{ m}$ from goal), the model was forbidden from calling `stop` at the destination.
     c) **Taxonomy Priority Bug:** In `failure_classifier.py`, `if oracle_success:` was evaluated before `if deadlock_terminated:`, mislabeling all 15 deadlock episodes as `stop_after_reach`.
2. **Empirical Stop Behavior in Isolation:**
   - In isolated tests placing the agent $0.55\text{ m}$ from the destination, lifting the artificial distance prohibition and providing informative obstacle reselect feedback caused `qwen3.5:4b` to call `stop` immediately with accurate landmark rationale: *"I have reached the kitchen area and am stopped at a counter or wall, which matches the instruction to stop near the counter in the kitchen."*
3. **Metric Implementation Sanity Confirmed:**
   - An oracle Shortest-Path Follower achieved **$\text{SR} = 100.0\%$ and $\text{SPL} = 100.0\%$** across all 15 episodes, proving that the geodesic distance, pathfinding, and metric calculations are strictly correct.
4. **Multi-Scene Verification Across 3 Distinct Matterport3D Meshes:**
   - Extended validation across three separate scenes (`gZ6f7yhEvPG`, `pLe4wQe7qrG`, `sT4fr6TAbpF`), with G4.1–G4.8 geometry checks passing 100% across all three.

---

## 2. Gated Verification Matrix (Phase 0 to Phase 10)

| Phase | Title | Brief & Gate Requirement | Evidence Path | Gate Status |
| :--- | :--- | :--- | :--- | :--- |
| **0** | **Source of Truth** | Spec ledger committed with 38 parameters mapped 1-to-1 to code constants; episode subset identified; target comparison defined. | [`audit/phase0/phase0_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase0/phase0_gate_summary.json)<br>[`spec/spec_ledger.yaml`](file:///c:/Users/Kruti/Desktop/PROG/UAV/spec/spec_ledger.yaml) | **PASS** |
| **1** | **Host Baseline** | GPU, driver, Ollama version, WSL2 connectivity, idle VRAM margins, and environment lockfiles pinned. | [`audit/phase1/phase1_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase1/phase1_gate_summary.json)<br>[`audit/phase1/environment_wsl_habitat.txt`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase1/environment_wsl_habitat.txt) | **PASS** |
| **2** | **Vision Model Layer** | 10-check battery on official `qwen3.5:4b` proving true multimodal vision (OCR, color, shape, count, FOV, 5-image payload, cold/warm latency <15s). | [`audit/phase2/phase2_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase2/phase2_gate_summary.json)<br>[`audit/phase2/qwen3.5_4b_battery.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase2/qwen3.5_4b_battery.json) | **PASS** |
| **3** | **Simulator Go/No-Go Gate** | Habitat-Sim 0.3.3 installed in WSL2; Matterport3D `gZ6f7yhEvPG` downloaded; hardware OpenGL 4.6 core profile verified on RTX 4050. | [`audit/phase3/phase3_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase3/phase3_gate_summary.json)<br>[`audit/phase3/g34_mp3d_gZ6f7yhEvPG_render.png`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase3/g34_mp3d_gZ6f7yhEvPG_render.png) | **PASS** |
| **4** | **Embodiment Adapter & Multi-Scene Geometry** | Verified across 3 real scenes (`gZ6f7yhEvPG`, `pLe4wQe7qrG`, `sT4fr6TAbpF`). Intrinsics ($fx=fy=cx=320, cy=240$), 360° turn loop closure ($0.00\text{ m}$ drift), forward step calibration ($0.2500\text{ m}$), 4-view panoramic continuity ($0.00\text{ px}$ diff), metric floor datum ($Y \in [-0.076, -0.048]\text{ m}$). | [`audit/phase4/multi_scene_gate_verification.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase4/multi_scene_gate_verification.json) | **PASS** |
| **5** | **Occupancy Grid & Memory** | 3-tier grid representation, log-odds updates ($l_{free}=-0.405, l_{occ}=0.847$), chassis height band filtering ($[0.10, 1.25]\text{ m}$), ray clearing, map expansion on real trajectory, recall tool validity. | [`audit/phase5/phase5_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase5/phase5_gate_summary.json)<br>[`audit/phase5/g54_real_mp3d_bev.png`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase5/g54_real_mp3d_bev.png) | **PASS** |
| **6** | **Deterministic Harness** | Bounds clipping, standoff subtraction, corridor collision testing with synthetic point clouds, budget enforcement ($E \le 4, R \le 5$), stop tool behavior, 49/49 pytest unit tests passed. | [`audit/phase6/phase6_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase6/phase6_gate_summary.json)<br>[`audit/phase6/g66_unit_tests.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase6/g66_unit_tests.json) | **PASS** |
| **7** | **Closed-Loop Tool Calling** | Valid JSON schema generation, coordinate bounds, active navigation tool selection, clean arrival stop tool issuance, latency 2.30s < 15s budget, closed-loop episodes on real Habitat-Sim. | [`audit/phase7/phase7_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase7/phase7_gate_summary.json)<br>[`audit/phase7/g76_habitat_final_bev.png`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase7/g76_habitat_final_bev.png) | **PASS** |
| **8** | **Single-Episode Provenance Audit** | Run manifest sealed before step 1; 1 raw RGB and 1 metric depth frame archived per step; full JSONL step trace; final BEV map generated; zero oracle leakage audit; metric integrity confirmed. | [`audit/phase8/phase8_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase8/phase8_gate_summary.json)<br>[`audit/phase8/manifest.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase8/manifest.json) | **PASS** |
| **9** | **Benchmark & Baselines Battery** | Baseline sanity comparison (Shortest-Path Follower, Stop-Immediately, Random Walker); Wilson score confidence intervals; start-to-goal distance ledger; root cause diagnosis. | [`audit/phase9/baseline_sanity_battery.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase9/baseline_sanity_battery.json)<br>[`audit/phase9/stop_isolation/multi_turn_stop_battery.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase9/stop_isolation/multi_turn_stop_battery.json) | **PASS** |
| **10** | **Config Freeze & Reorganization** | Regression smoke test passing (`scripts/regression_smoke.sh`); legacy AirSim/remote code archived; clean repository structure. | [`audit/phase10/phase10_gate_summary.json`](file:///c:/Users/Kruti/Desktop/PROG/UAV/audit/phase10/phase10_gate_summary.json) | **PASS** |

---

## 3. Baseline Sanity Battery & Statistical Comparison (PRE-FIX, SUPERSEDED)

> The table below used the synthetic `episode_list_100.json`. It is not an R2R-CE
> result and must not be cited. Only Section 0.2 is valid.

### 3.1 Local Baselines on Benchmark Split ($N = 15$ Episodes in `gZ6f7yhEvPG`)

To verify metric calculation integrity and assess the difficulty of the 15-episode test set, three reference baselines were executed against the identical Habitat-Sim environment and test episodes:

| Agent / Model | SR (%) | SPL (%) | OSR (%) | Mean NE (m) | Mean Trajectory (m) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Shortest-Path Oracle Follower** | **100.0%** | **100.0%** | **100.0%** | **0.86 m** | **2.67 m** | Perfect execution; proves geodesic metric math is completely sound. |
| **Stop-Immediately Agent** | **66.7%** | **66.7%** | **66.7%** | **2.70 m** | **0.00 m** | 10 of 15 episodes start at $\le 3.0\text{ m}$ from goal. |
| **Random Walker (25 actions)** | **33.3%** | **10.8%** | **93.3%** | **1.96 m** | **4.77 m** | High OSR confirms that random motion easily enters the 3m bubble in compact indoor scenes. |
| **AgenticNav VLM (`qwen3.5:4b`)** | **0.0%** | **0.0%** | **100.0%** | **2.33 m** | **5.37 m** | Reached goal in 15/15, but deadlocked at terminal obstacle due to harness standoff policy. |

### 3.2 Statistical Confidence Intervals (Wilson Score Interval)

On $N = 15$ episodes, empirical bootstrap intervals $[100\%, 100\%]$ were degenerate. The true statistical bounds (95% Wilson score intervals) are:

- **Oracle Success Rate (OSR):** $100.0\%$ (15/15) $\implies$ **[79.6%, 100.0%]**
- **Success Rate (SR):** $0.0\%$ (0/15) $\implies$ **[0.0%, 20.4%]**

### 3.3 Start-to-Goal Geodesic Distance Ledger

The 15 test episodes have a mean start-to-goal distance of only **$2.70\text{ m}$**:
- **Episodes starting $< 3.0\text{ m}$ (10 episodes):** `ep_0002` (2.47m), `ep_0003` (2.17m), `ep_0004` (2.47m), `ep_0005` (2.47m), `ep_0006` (2.47m), `ep_0009` (2.57m), `ep_0010` (2.57m), `ep_0011` (2.57m), `ep_0012` (2.47m), `ep_0013` (2.17m).
- **Episodes starting $> 3.0\text{ m}$ (5 episodes):** `ep_0001` (3.28m), `ep_0007` (4.16m), `ep_0008` (3.28m), `ep_0014` (4.16m), `ep_0015` (3.28m).

This explains why OSR was 100% for the VLM and 93.3% for the random walker: in compact indoor floorplans, short routes naturally cross within the standard 3-meter success threshold.

---

## 4. Root-Cause Analysis: Resolving the SR 0% vs OSR 100% Anomaly

### 4.1 The Standoff & Quantization Deadlock Trap
When the agent approaches a target landmark (e.g. bed, table, counter, doorway) to within $D_{\text{raw}} = 0.50\text{ m}$:
1. The standoff distance $m_{\text{eff}} = 0.30\text{ m}$ is subtracted: $\Delta d = 0.50 - 0.30 = 0.20\text{ m}$.
2. Forward primitive steps have a discrete resolution of $0.25\text{ m}$.
3. Quantization rounds $0.20\text{ m} < 0.25\text{ m}$ down to $\rho_q = 0.0\text{ m}$.
4. `GroundCorridorChecker.evaluate()` returns `{"status": "reselect", "cause": "no_motion"}`.
5. In the legacy harness, no explanatory guidance was provided to the model. The model repeated the forward selection 5 times consecutively, exhausting $R_{max}=5$ and aborting as `deadlock` without calling `stop`.

### 4.2 The Artificial Prompt Prohibition
The legacy prompt contained the following rule:
> *"You must NEVER call stop after only 1-3 moves or under 6.0 meters of travel. If you face a wall or low clearance, you must TURN, NOT stop."*

Because 100% of the benchmark episodes had start-to-goal lengths under 4.2m, the model was explicitly forbidden by prompt instruction from calling `stop`, even when standing directly in front of the goal destination.

### 4.3 Taxonomy Priority Bug
In `agenticnav_paper_replica/evaluation/failure_classifier.py`, the waterfall was:
```python
# OLD BUGGY ORDER:
if oracle_success:
    return "stop_after_reach"  # Eclipsed deadlock_terminated!
if deadlock_terminated:
    return "safety_deadlock"
```
Because the agent entered the 3m bubble in all 15 episodes, `if oracle_success:` triggered first and labeled all episodes as `stop_after_reach`, masking the fact that `called_stop` was `False` and that termination occurred due to deadlock.

The classifier has now been re-ordered:
```python
# FIXED STRICT WATERFALL:
if infrastructure_error:
    return "infrastructure_failure"
if deadlock_terminated:
    return "safety_deadlock"
if called_stop:
    return "stop_after_reach" if oracle_success else "stop_premature"
if protocol_violation:
    return "protocol_violation"
if oracle_success:
    return "missed_stop"
if action_budget_exceeded:
    return "budget_exceeded"
return "unclassified"
```

### 4.4 Empirical Proof: Stop Behavior in Isolation (Check 2)
To prove that `qwen3.5:4b` has genuine stopping capability when given proper guidance, the agent was placed $0.55\text{ m}$ from the destination in Habitat-Sim across multiple test conditions:

1. **Condition 1 (Legacy prompt with <6m prohibition):** Agent refused to stop; called `query_depth` and then attempted forward `move_to`.
2. **Condition 2 (Clean milestone prompt, no distance prohibition):** Agent actively queried depth in front view ($0.67\text{ m}$ clearance) and attempted to reach the landmark.
3. **Condition 3 (Clean prompt with thinking=True):** Deep thinking chain spent 30.4s evaluating the visual scene and correctly reasoned about orientation.
4. **Condition 4 (Milestone completed feedback):** Agent called **`stop`** immediately (latency 3.89s) with exact rationale:
   > *"Goal 'Exit the room and turn left. Walk forward towards the kitchen and stop near the counter' is complete. I am currently in the hallway, which corresponds to the location of the kitchen counter as indicated by the navigation instruction. All milestones have been satisfied."*
5. **Multi-Turn Reselect Feedback:** When facing a terminal counter/wall and receiving the obstacle feedback:
   > `{"status": "reselect", "cause": "no_motion", "message": "Forward motion blocked by counter/wall. If this is your destination, call stop(). Otherwise turn."}`
   The model invoked **`stop`** within 3.72s with rationale:
   > *"I have reached the kitchen area and am stopped at a counter or wall, which matches the instruction to stop near the counter in the kitchen."*

---

## 5. Multi-Scene Gate Verification (Check 3)

The G4 embodiment geometry suite was expanded from a single scene (`gZ6f7yhEvPG`) to **three distinct continuous Matterport3D scenes** downloaded directly from Hugging Face:
1. `gZ6f7yhEvPG`
2. `pLe4wQe7qrG`
3. `sT4fr6TAbpF`

| Check | Specification | `gZ6f7yhEvPG` | `pLe4wQe7qrG` | `sT4fr6TAbpF` | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **G4.3 Loop Closure** | 24 $\times$ 15° turns $\implies$ position drift $< 1\text{ mm}$, rotation drift $< 0.05^\circ$ | Drift: $0.00\text{ m}$, $0.00^\circ$ | Drift: $0.00\text{ m}$, $0.00^\circ$ | Drift: $0.00\text{ m}$, $0.00^\circ$ | **PASS** |
| **G4.4 Forward Step** | Nominal $0.25\text{ m}$ displacement error $< 1\text{ mm}$ | Measured: $0.2500\text{ m}$ | Measured: $0.2500\text{ m}$ | Measured: $0.2500\text{ m}$ | **PASS** |
| **G4.5 Visual Continuity** | 90° right turn View 0 vs View 1 pixel difference $< 15\text{ px}$ | Mean diff: $0.00\text{ px}$ | Mean diff: $0.00\text{ px}$ | Mean diff: $0.00\text{ px}$ | **PASS** |
| **G4.6 Floor Datum** | Unprojected floor points within $\pm 0.12\text{ m}$ of $Y=0.0\text{ m}$ floor plane | Mean $Y = -0.076\text{ m}$ | Mean $Y = -0.069\text{ m}$ | Mean $Y = -0.048\text{ m}$ | **PASS** |

All three scenes passed 100% of geometric, optical, and kinematic requirements.

---

## 6. Repository Architecture

The codebase has been refactored and organized into an audited, production-grade layout. All legacy AirSim UAV files, Docker scripts, and loose testing files have been archived:

```
PROG/UAV/
├── agenticnav_paper_replica/       # Core replication codebase
│   ├── configs/                   # Spec ledger configurations (ReplicaConfig)
│   ├── data/                      # Episode manifests (100 episodes) & splits
│   ├── embodiment/                # HabitatGroundEmbodiment (Habitat-Sim 0.3.3)
│   ├── evaluation/                # Metrics & Waterfall Failure Classifier
│   ├── geometry/                  # Pinhole transforms & corridor checking
│   ├── harness/                   # StepDecisionLoop, ToolExecutor, Prompts
│   ├── mapping/                   # OccupancyGrid2D & BEVRenderer
│   ├── memory/                    # EpisodicImageCache & ActionHistory
│   ├── traces/                    # JSONL EpisodeTrace logging
│   └── vlm/                       # OllamaNavClient (local GPU offload)
├── audit/                         # Immutable evidence folders per phase (phase0 - phase10)
│   ├── phase3/                    # Hardware render PNGs & context logs
│   ├── phase4/                    # Multi-scene geometry verification
│   └── phase9/                    # Baseline sanity, stop isolation, benchmarks
├── spec/                          # Ground-truth spec_ledger.yaml
├── scripts/                       # Verification batteries, baselines, smoke tests
└── archive/                       # Archived legacy AirSim, UAV, and remote cluster scripts
```

---

## 7. Next Steps & Recommendations (POST-FIX)

1. **Do not cite the pre-fix phase9 numbers.** Use the official 17-episode run in
   `audit/phase9_postfix/` for any claim about the current ground replica.
2. **Diagnose the zero-action failures.** The dominant failure is `safety_deadlock`
   with 0 simulator actions; inspect whether the VLM targets above-horizon pixels or
   whether stair risers should be excluded from the 2D obstacle slice. Any change
   must be recorded as a new ablation, not silently folded into the baseline.
3. **Model capacity check.** `qwen3.5:4b` frequently ignores the "floor point"
   constraint. Compare against a stronger local vision model before concluding the
   harness itself is at fault.
4. **Full-split extension.** The current result is 17 episodes across 3 locally
   available scenes, not the full 1,839-episode `val_unseen` split.
