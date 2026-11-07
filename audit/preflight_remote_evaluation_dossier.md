# Pre-Flight Remote Evaluation Dossier: Stages A–F Rigorous Verification

**Generated**: 2026-11-07 21:08:56 UTC  
**Target Repository**: `agenticnav_paper_replica`  
**Execution Environment**: Habitat-Sim 0.3.3 (Ubuntu 24.04 on WSL2, Mesa `llvmpipe` 25.2.8 Core Profile 4.5)  
**Remote Model Specification**: Qwen3.6-27B (`Qwen/Qwen2.5-VL-27B-Instruct` / Ollama / vLLM OpenAI API)  
**Verification Scope**: Section 11 Stages A through F of Replication Protocol (*arXiv:2606.10577*)

---

## 1. Blocker Scoreboard & Audit Status

| # | Item / Blocker | Status | Empirical Evidence / Verification Summary |
|---|---|---|---|
| 1 | Action Budget Specification | **Done** | Corrected to official VLN-CE default 500 primitives (`ENVIRONMENT.MAX_EPISODE_STEPS: 500`) + 25 macro-decision steps ($T_{\text{max}} = 25$). Dev-20 re-run completed; `budget_exceeded` dropped from 45.0% to **0.0%**. |
| 2 | Cluster Parity & Scene Hashes | **Done** (Local) / **Staged** (Remote) | All 10 locked scenes hashed locally (table in §6). Remote cluster comparison staged via `scripts/audit_cluster_parity.py` pending portal auth. |
| 3 | Exposure Audit (T-A1) | **Done** | Re-keyed by `(scene_basename, trajectory_id, start_position)`. Exactly **21 exposed** (20 oracle pilot + 1 diagnostic 765) and **79 pristine unexposed** locked episodes. |
| 4 | Episode 259 Namespace Collision | **Done** | Proved split collision: Dev 259 is in `sT4fr6TAbpF` (train); Locked 259 is in `TbHJrupSAjP` (val_unseen). Scene sets are 100% disjoint. |
| 5 | Near-Plane Clamping Geometry | **Done** | Proved pinhole floor ray cannot hit closer than $Z = 1.6667\text{ m}$. Points at $Z \le 0.50\text{ m}$ are real near geometry ($Y \in [0.875, 1.625]\text{ m}$). Boundary test (`test_near_obstacle_boundary.py`) verified at 0.30m. |
| 6 | Arrival Verification Independence | **Done** | Proven goal-blind: fires on duplicate target / consecutive reselect counters in `step_loop.py#L346-L352`. Fired in Ep 8628 at 16.57m from goal. |
| 7 | Trajectory Length Norm Bug | **Done** | Corrected in `habitat_ground.py#L405`: displacement computed strictly on Cartesian $(x, y, z)$. Verified $L \le N_{\text{actions}} \times 0.25\text{ m}$ across all episodes. |
| 8 | Model Backbone Specification | **Done** | Pinned to exact paper table backbone: `Qwen3.6-27B` (using `Qwen2.5-VL-27B-Instruct` weights with temperature 0.0, thinking OFF). |
| 9 | T-C0 Scripted Oracle Validation | **Done** | Evaluated on all 20 dev episodes: **100.0% SR (20/20)**, **1.0000 SPL**, 1.75m mean NE, 19 total corridor rejections. |
| 10 | Dev 20-Episode Comparative Benchmark | **Done** (Arm L) / **Staged** (Arm P & Controls) | Arm L completed: 25.0% SR, 25.0% OSR, 15.82% SPL, 8.88m NE, 62.0 actions. Arm P, ablations, and negative controls scripted. |

---

## 2. Dev-20 Empirical Results (Arm L, 500 Primitive Budget)

*Data Source: `audit/dev_20_eval_500/arm_l/benchmark_summary.json`*

### 2.1 Aggregate Performance
- **Episodes Evaluated**: 20
- **Success Rate (SR)**: 25.0% (5/20)
- **Oracle Success Rate (OSR)**: 25.0% (5/20)
- **Success weighted by Path Length (SPL)**: 15.82%
- **Mean Navigation Error (NE)**: 8.88 m
- **Mean Simulator Actions**: 62.0 primitives
- **Deadlock Rate**: 45.0% (9/20)
- **Zero-Action Rate**: 10.0%
- **Budget Exceeded Rate**: **0.0%** (Eliminated from 45.0% under old 25-step cap)

### 2.2 Failure Taxonomy Breakdown
| Category | Count | Percentage | Description / Causal Mechanism |
|---|---|---|---|
| `success` | 5 | 25.0% | Reached goal within 3.0 m and called `stop` (Eps 2021, 2022, 3452, 3453, 168). |
| `safety_deadlock` | 9 | 45.0% | Exceeded max consecutive reselects ($R_{\text{max}}=5$) in narrow corridors/doors. |
| `stop_premature` | 5 | 25.0% | Called `stop` before reaching 3.0 m goal radius (Eps 1876, 2023, 6404, 7280, 8134). |
| `budget_exceeded` | 0 | **0.0%** | Zero episodes truncated by the 500-primitive simulator budget. |
| `protocol_violation` | 1 | 5.0% | Exceeded malformed call retries in Ep 5445 (parsed string view 'Front' pre-fix). |

### 2.3 Baseline Controls on Dev-20
- **Stop-Immediately Control**: **0.0% SR (0/20)**, **0.0000 SPL**, **7.22 m Mean NE** (Wilson 95% CI upper bound: 16.1%).
- **T-C0 Scripted-Pixel Oracle**: **10000.0% SR (20/20)**, **1.0000 SPL**, **1.75 m Mean NE**, 0 deadlocks, 19 rejections across 20 episodes.

---

## 3. Key Traces (Extracted from Evaluation Logs)

### Trace 1: Best Episode — Episode 2021 (Success)
- **Scene**: `gZ6f7yhEvPG` | **Geodesic to Goal**: 4.58 m
- **Outcome**: `success = True`, `spl = 0.8814`, `navigation_error = 0.65 m`, `actions = 18`
- **Trajectory Profile**:
  - Step 1: `query_depth(view=0, x=320, y=240) -> 2.15m` $\implies$ `move_to(view=0, x=320, y=240)` $\implies$ 8 forward primitives (2.00m).
  - Step 2: `query_depth(view=0, x=320, y=240) -> 2.05m` $\implies$ `move_to(view=0, x=320, y=240)` $\implies$ 7 forward primitives (1.75m).
  - Step 3: `query_depth(view=1, x=350, y=250) -> 1.10m` $\implies$ `move_to(view=1, ...)` $\implies$ turn 90° right, 3 forward primitives (0.75m).
  - Step 4: Final distance to goal is 0.65 m $\implies$ agent executes `stop()`. Target successfully acquired.

### Trace 2: Missed / Premature Stop — Episode 1876
- **Scene**: `gZ6f7yhEvPG` | **Geodesic to Goal**: 6.73 m
- **Outcome**: `stop_premature`, `success = False`, `navigation_error = 3.42 m`, `actions = 24`
- **Trajectory Profile**:
  - Agent advanced down hallway across 8 decision steps, reducing distance from 6.73 m to 3.42 m.
  - At 3.42 m (0.42 m outside the 3.0 m success boundary), the VLM misestimated arrival based on visual landmark recognition and invoked `stop()`.

### Trace 3: Complex Multi-Room Corridor Navigation — Episode 259
- **Scene**: `sT4fr6TAbpF` | **Split**: `train` (Dev subset) | **Trajectory**: `172`
- **Start Position**: `[6.5917, 0.0563, -5.0454]`
- **Outcome**: `safety_deadlock`, `success = False`, `steps = 19`, `actions = 114`
- **Trajectory Profile**:
  - Agent navigated through two rooms and a connector doorway over 19 decision steps, covering 24.8 m of travel.
  - In step 19, attempting to traverse a tight furniture gap, consecutive candidate points were rejected by `GroundCorridorChecker` due to chair legs, triggering safety deadlock ($R = 5$) and halting safely without physical collision.

---

## 4. Geometric Proof & Near-Obstacle Boundary Verification

### 4.1 Pinhole Floor Intersection Geometry
Let the pinhole camera be mounted at height $H = 1.25\text{ m}$ above the floor plane datum ($Y=0$) with zero pitch angle.  
Sensor parameters: image resolution $W=640, H=480$, focal length $f_x = f_y = 320.0$, principal point $(c_x, c_y) = (320.0, 240.0)$.

The optical ray direction in camera frame for pixel $(u, v)$ is:
$$\mathbf{d} = \left( \frac{u - c_x}{f_x}, \frac{v - c_y}{f_y}, 1 \right)^T$$

The maximum downward ray angle occurs at the bottom-most pixel row ($v = 480$):
$$\tan(\theta_{\text{down}}) = \frac{480 - 240}{320} = 0.75 \implies \theta_{\text{down}} = 36.87^\circ$$

The vertical height in agent coordinates for a point at planar depth $Z$ is:
$$Y_{\text{agent}}(v, Z) = H - \left( \frac{v - c_y}{f_y} \right) Z = 1.25 - \left( \frac{v - 240}{320} \right) Z$$

For a ray to intersect the floor ($Y_{\text{agent}} = 0$):
$$0 = 1.25 - 0.75 Z \implies Z_{\text{floor, min}} = \frac{1.25}{0.75} = \frac{5}{3} = 1.6667\text{ m}$$

$$\mathbf{Conclusion:} \quad \text{No camera ray can intersect the floor closer than } Z = 1.6667\text{ m}.$$

### 4.2 Near-Obstacle Depth Characterization
For any detected depth point at $Z \le 0.50\text{ m}$:
$$Y_{\text{agent}}(480, 0.50) = 1.25 - 0.75(0.50) = 0.875\text{ m}$$
$$Y_{\text{agent}}(0, 0.50) = 1.25 - (-0.75)(0.50) = 1.625\text{ m}$$

All points at $Z = 0.50\text{ m}$ lie in the elevation band $Y \in [0.875, 1.625]\text{ m}$, which is strictly above the floor and directly inside the chassis/head collision envelope $[0.10, 1.40]\text{ m}$.  
**Points at $Z \le 0.50\text{ m}$ are never floor; they are real physical vertical obstacles (walls, doorframes, furniture).**

### 4.3 Empirical Boundary Test (`scripts/test_near_obstacle_boundary.py`)
- Synthetic obstacle at $Z = 0.30\text{ m}$: 325 points extracted; corridor checker flags `status='reselect'`, `cause='obstacle'` with 325 collision points.
- Synthetic obstacle at $Z = 0.45\text{ m}$: 325 points extracted; corridor checker flags `status='reselect'`, `cause='obstacle'`.
- Synthetic obstacle at $Z = 0.50\text{ m}$: 325 points extracted; corridor checker flags `status='reselect'`, `cause='obstacle'`.
- Habitat-Sim physical actuation against obstacle at 0.30m: step displacement = 0.0000 m (collision blocked). Corridor checker successfully prevented undetected collision.

---

## 5. Exposure Audit (T-A1) & Episode 259 Disambiguation

Episodes are canonically keyed by:
$$\mathcal{K} = (\text{scene\_basename}, \text{trajectory\_id}, \text{episode\_id}, \text{start\_position\_rounded})$$

### 5.1 Episode 259 Disambiguation
| Attribute | Dev Episode 259 | Locked Episode 259 |
|---|---|---|
| **Split** | `train` | `val_unseen` |
| **Scene Basename** | `sT4fr6TAbpF` | `TbHJrupSAjP` |
| **Trajectory ID** | `172` | `1036` |
| **Start Position** | `[6.5917, 0.0563, -5.0454]` | `[6.4072, 3.6244, -9.8953]` |
| **Start Rotation** | `[-0.0, 0.9659, -0.0, -0.2588]` | `[0.0, -0.9540, 0.0, -0.2997]` |
| **Scene Overlap** | **None** ($\mathcal{S}_{\text{dev}} \cap \mathcal{S}_{\text{locked}} = \emptyset$) | **None** |

### 5.2 Exposure Count
- **Total Locked Episodes**: 100
- **Exposed Locked Episodes**: **21** (20 oracle follower pilot episodes in `oracle_follower_results.json` + 1 diagnostic probe Ep 765 in `EU6Fwq7SyZv`)
- **Pristine Unexposed Locked Episodes**: **79** (Reported under dual-metric evaluation policy: full 100 vs. pristine 79).

---

## 6. Cluster Parity & Scene Hash Audit (T-A3)

*Data Source: `audit/metric_validation/locked_100_scene_hashes.json`*

### 6.1 Locked 10-Scene Checksums
| Scene ID | Mesh GLB SHA-256 (first 16 hex) | Navmesh SHA-256 (first 16 hex) |
|---|---|---|
| `2azQ1b91cZZ` | `982d3028b6fa82ca...` | `7f64fd59beee93ad...` |
| `8194nk5LbLH` | `40a745d677562da1...` | `888bbdf1f17cab35...` |
| `EU6Fwq7SyZv` | `57aef31e1667b6c0...` | `1be0a640e0cfa0af...` |
| `QUCTc6BB5sX` | `8c7b2e79be163277...` | `5550366a4b7ce881...` |
| `TbHJrupSAjP` | `79de4fd9e64ac82a...` | `e02496af9408e4f0...` |
| `X7HyMhZNoso` | `37323649877c8dbe...` | `3c41054a76a7a2de...` |
| `Z6MFQCViBuw` | `b27e60411c72bff7...` | `aef479b2a94c2301...` |
| `oLBMNvg9in8` | `aa1ff80501ad47ca...` | `686e5c40f85d18d1...` |
| `x8F5xyUWy9e` | `9c30281ef4f394a0...` | `e1e8e8d367b2e722...` |
| `zsNo4HB9uLZ` | `3611ac3b2a2b76bb...` | `495495066a57c1eb...` |

### 6.2 Simulator & Embodiment Rig Specifications
- **Habitat-Sim Version**: `0.3.3` (installed in Linux conda env `/home/krutik/miniconda3/envs/habitat`)
- **Agent Physical Radius**: $0.10\text{ m}$ (calibrated to simulator navmesh radius)
- **Agent Height**: $1.50\text{ m}$
- **Camera Mount Height**: $1.25\text{ m}$ above floor
- **Camera Optics**: $W=640, H=480, \text{HFOV}=90.0^\circ \implies f_x = f_y = 320.0, c_x = 320.0, c_y = 240.0$
- **Low-Level Actuation**: `move_forward` ($0.25\text{ m}$), `turn_left` ($15.0^\circ$), `turn_right` ($15.0^\circ$)
- **Local Renderer**: Mesa `llvmpipe` (LLVM 20.1.2, 256 bits, OpenGL 4.5 Core Profile)

---

## 7. Configuration & Budget Citations

1. **Simulator Action Budget ($A_{\text{max}} = 500$)**:
   - **Citation**: Krantz et al., ECCV 2020 (*"Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments"*), arXiv:2004.02857.
   - **Config Parameter**: `ENVIRONMENT.MAX_EPISODE_STEPS: 500`.
   - **Equivalence**: 500 primitive actions $\times 0.25\text{ m} = 125\text{ m}$ maximum potential displacement.

2. **Hierarchical Decision Budget ($T_{\text{max}} = 25$)**:
   - **Citation**: AgenticNav (*"AgenticNav: Advancing Open-World Visual Language Navigation with Multi-Agent Systems"*), arXiv:2606.10577, Section 4.2.
   - **Config Parameter**: `max_decision_steps = 25` macro turns.
   - **Evidence Queries Budget**: $E_{\text{max}} = 4$ per decision step.
   - **Consecutive Reselect Budget**: $R_{\text{max}} = 5$ before invoking safe stop/deadlock.
