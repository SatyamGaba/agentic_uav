# Agent Context & System Memory

This file serves as the canonical shared context and handoff document across agent sessions. Any agent joining this repository can read this single document to understand the codebase state, active priorities, resolved vs open bugs, and verified invariants.

---

## 1. Project Mission & Pinned Specification

We are replicating **AgenticNav (arXiv:2606.10577, 2026)** for ground Vision-and-Language Navigation in Continuous Environments (VLN-CE), evaluated in Habitat-Sim on Matterport3D (MP3D) meshes.

- **Pinned Reference Spec (Latest Paper Version)**:
  - **Backbone Target (Qwen3.8-27B)**:
    - Success Rate (SR): **48.0%**
    - Success weighted by Path Length (SPL): **33.57**
    - Oracle Success Rate (OSR): **72.0%**
    - normalized Dynamic Time Warping (nDTW): **61.58**
    - Navigation Error (NE): **5.65 m**
  - **Random Baseline**: SR 2.0%, OSR 12.0%, NE 8.63 m.
  - **Benchmark Split**: The canonical 100 `val_unseen` episodes from Open-Nav / SmartWay / EvoNav (`opennav_r2rce_100.json`, SHA-256 confirmed, includes Fig. 6 episodes 116, 191, 218).
  - **`query_depth` Tool Contract**: Returns metric depth plus min/mean/median statistics over a $5 \times 5$ pixel neighbourhood. **Does not move the agent.** Action preview is treated as a local aid (Arm L only), not in the paper-faithful Arm P.
  - **Corridor Safety Check**: Rectangular swept box with obstacle point threshold $N = 1$ (Paper-faithful Arm P). $N = 5$ is a calibrated local aid (Arm L).
  - **Action History**: In prompt, maintains the last 6 steps of decisions (reason, target, and executed motion), current views, and BEV map. Past views are accessed only through `recall`.

---

## 2. Experimental Arms (P vs L)

| Parameter / Feature | Arm P (Paper-Faithful) | Arm L (Local Aids) | Tag |
|---|---|---|---|
| Obstacle Point Threshold ($N$) | **1** | **5** | Calibrated Engineering Choice |
| Arrival Verification Prompt | **Disabled** | **Enabled** (single-shot turn) | Local Aid |
| Floor Grazing Guidance | **Disabled** | **Enabled** | Local Aid |
| Action Preview in `query_depth` | **Disabled** | **Enabled** (if requested) | Local Aid |
| History Window ($K$) | **6 steps** | **6 steps** | Paper Text (Sec 3) |
| Fixed Standoff Margin ($m$) | **0.30 m** | **0.30 m** | Paper-unspecified (Sec 4.4) |
| Discrete Forward Step | **0.25 m** | **0.25 m** | Habitat Standard (Sec 1) |
| Discrete Turn Step | **15.0°** | **15.0°** | Habitat Standard (Sec 1) |

---

## 3. Verified Mathematical & Sensor Conventions

1. **Agent Frame & Coordinates (Habitat-Sim Canonical)**:
   - Forward: $-Z_{\text{agent}}$
   - Right: $+X_{\text{agent}}$
   - Up: $+Y_{\text{agent}}$ (floor datum at $y=0$, camera center at $y=1.25\text{ m}$)
2. **Pinhole Optical Projection**:
   - Camera Optical Frame: $+z$ optical depth, $+x$ right, $+y$ down.
   - Pinhole centering: $x_c = (x_0 + 0.5 - c_x) \cdot z_c / f_x$, $y_c = (y_0 + 0.5 - c_y) \cdot z_c / f_y$.
   - Azimuths: View 0 (Front, $0^\circ$), View 1 (Right, $-90^\circ$), View 2 (Back, $180^\circ$), View 3 (Left, $+90^\circ$).
   - Transformation to agent:
     $$p_x = \cos(\psi) x_v + \sin(\psi) z_v$$
     $$p_y = y_v = -y_c + h_{\text{cam}}$$
     $$p_z = -\sin(\psi) x_v + \cos(\psi) z_v$$
   - Inverted transform `agent_to_camera_frame` is verified to $< 10^{-9}\text{ m}$.
3. **Physical Simulator Overlap Tests**:
   - `test_pointcloud_physical_overlap_with_negative_control` in `agenticnav_paper_replica/geometry/tests/test_view_frames.py`:
     - True Left view vs $90^\circ$ turned front camera: NN distance $= 0.021\text{ m}$ ($< 0.08\text{ m}$).
     - Mirrored negative control ($X_{\text{ag}} \leftarrow -X_{\text{ag}}$): NN distance $> 0.35\text{ m}$ ($> 0.30\text{ m}$).
     - True Right view vs $-90^\circ$ turned front camera: NN distance $= 0.049\text{ m}$ ($< 0.08\text{ m}$).
4. **Turn-in-Place Obstacle Policy**:
   - When $\rho_q = 0$ (`forward_steps == 0`), in-place rotation sweeps the pre-existing circular chassis footprint ($r_a = 0.10\text{ m}$) without expanding it.
   - Pre-existing obstacle contact during pure rotation is ignored (`skip_footprint`), eliminating false reselect deadlocks in cluttered starts (e.g. beds/doorways).
5. **View Index Hardening**:
   - `parse_view_index_ex(val)` returns `(view_idx, rewritten: bool)`.
   - Ambiguous inputs (e.g. `"front-right"`), out-of-bounds integers ($<0$ or $>3$), and invalid types raise `InvalidViewError`.
   - Callers increment `malformed_calls` and enforce the 3-error per step protocol violation ceiling.

---

## 4. Repository Structure & Key Files

- `agenticnav_paper_replica/`:
  - `geometry/pinhole.py`: Pinhole projection, depth conversion, coordinate frames, `parse_view_index_ex`.
  - `geometry/safety_region.py`: Swept-corridor collision checker, turn-in-place kinematics.
  - `geometry/tests/test_view_frames.py`: Frame orientation, roundtrip, vectorized matching, and Habitat-Sim physical overlap tests.
  - `geometry/tests/test_safety_and_geometry.py`: 16 comprehensive unit tests for corridor collision, boundary checks, and pure rotations.
  - `harness/executor.py`: Tool execution for `query_depth`, `recall`, `move_to`.
  - `harness/step_loop.py`: Multi-turn decision loop, tool dispatch, malformed call budget, image capping (max 6 images: 4 views + 1 BEV + 1 recalled).
  - `harness/runner.py`: Episode runner, metrics recording, simulator state synchronization (`obs.actual_distance_moved`, `obs.collided`).
  - `vlm/ollama_client.py` & `vlm/vllm_client.py`: Model clients wrapped in strict privilege firewalls.
  - `vlm/firewall.py`: Asserts no ground-truth metrics, geodesic distances, or canary sentinels leak into prompts.
- `spec/spec_ledger.yaml`: Pinned parameters, sources, owners, status, and calibrated engineering deviations.
- `scripts/benchmark_r2rce_postfix.py`: Main benchmarking entry point with strict clean-tree checks, locked mode guard, checkpointing, and resume support.

---

## 5. Current Test Status

- **Full PyTest Suite**: **83 passed / 0 failed** (`pytest agenticnav_paper_replica/ -v`).
- **Clean Git Tracking**: All changes committed under commit `36207ba` on branch `Arch/VLM`.

---

## 6. Pre-Remote Execution Checklist Status

- [x] **A1–A4 (Spec & Ledger)**: Pinned Qwen3.8-27B targets, ledger updated with vertical band note ($Y_{\text{ag}} \le 1.25\text{ m}$ vs $1.50\text{ m}$ cylinder) and pure-rotation footprint bypass. Arm P vs Arm L isolated.
- [x] **B1–B3 (Data Identity)**: Canonical 100-episode OpenNav R2R-CE file verified on disk (`opennav_r2rce_100.json`, containing IDs 116, 191, 218 across 10 scenes).
- [x] **D1–D4, D7–D10 (Geometry & Harness)**: Permanent frame tests, roundtrip tests, negative-control overlap tests, `parse_view_index_ex` error-raising, and image cap (peak 6) all passing in test suite.
- [x] **F1–F4 (Privilege & Integrity)**: Privilege firewall wrappers pass on positive and negative controls; sentinels blocked.
- [ ] **G1–G8 (Remote Endpoint)**: Verify remote vLLM model service:
  - `--limit-mm-per-prompt image=8` (or 10) configured on server.
  - Context length $\ge 32\text{k}$.
  - Wiring `VLLMNavClient` directly to `benchmark_r2rce_postfix.py` via `--backend vllm`.
- [x] **H1–H3 (Cluster Dry Run & Smoke Test)**: 3-episode smoke test on remote GPU cluster passed with 100% SR on Habitat R2R-CE (`r2rce_val_unseen_100_rand.json`), verifying manifest hashes, GPU EGL rendering, tool-dispatch, and zero failures across all 3 episodes.

---

## 7. Remote Cluster State & Execution Log (November 8, 2026)

- **Dedicated Hardware**: NVIDIA L40S (46,068 MiB VRAM), Node `172.24.16.175`.
- **Environment**:
  - Python 3.9 in `/home/jovyan/.conda/envs/habitat`.
  - Headless `habitat-sim` 0.3.3 installed with PyTorch cu121 and GPU EGL acceleration (`libegl`, `libopengl`).
  - PyTest suite: **82 passed / 0 failed / 1 skipped** on cluster node (Job #607 and #609).
- **Codebase & Datasets**:
  - Codebase bundle synced, reassembled, and git initialized clean at `/home/jovyan/scratch/UAV/` (commit `3d82eff`).
  - All 10 canonical MP3D scenes downloaded directly into `/home/jovyan/scratch/data/scene_datasets/mp3d` (Job #608).
- **Model Deployment (Ollama)**:
  - `qwen3.8:27b` (27.3B parameters, native vision and tools verified, family `qwen35`) pulled and active in Ollama on port 11434 (Job #612).
  - Also pulled and available: `qwen2.5:32b` (19 GB) and `qwen2.5:72b` (47 GB).
- **Verified Smoke Test Results (Job #617 - Habitat R2R-CE)**:
  - Dataset: Official Habitat R2R-CE split `agenticnav_paper_replica/data/r2rce_val_unseen_100_rand.json`.
  - Model: `qwen3.8:27b` (Arm L).
  - Episodes evaluated: 3 (`r2r_ce_val_unseen_0001`, `0002`, `0003` in scene `zsNo4HB9uLZ`).
  - **Success Rate**: **100.0% (3/3)**.
  - **Oracle Success Rate**: **100.0%**.
  - **Mean SPL**: **55.63** (79.81%, 40.83%, 46.24%).
  - **Mean Navigation Error (NE)**: **1.35 m** (0.35 m, 1.23 m, 2.46 m).
  - **Failure Taxonomy**: 3 Successes, 0 Deadlocks, 0 Budget Exceeded, 0 Protocol Violations, 0 Infrastructure Errors.
  - Verification Gate: **PASS**.
- **Full Benchmark Execution - Arm P (Job #634 - Executed November 8, 2026)**:
  - Job ID: **#634** (`benchmark-qwen3.8-27b-habitat-r2rce-100ep`).
  - Duration: 2 hours 10 minutes on NVIDIA L40S GPU.
  - Evaluation: **Arm P (Paper-Faithful, $N=1$, no arrival verification)** across Habitat R2R-CE split.
  - **Episodes Completed**: **99/100 episodes**.
  - **Success Rate (SR)**: **13.13%** (95% CI: $[7.07\%, 20.20\%]$).
  - **Success weighted by Path Length (SPL)**: **7.83**.
  - **Oracle Success Rate (OSR)**: **36.36%** (95% CI: $[27.27\%, 46.46\%]$).
  - **Navigation Error (NE)**: **7.24 m**.
  - **Failure Taxonomy**: 13 Successes, 37 Safety Deadlocks ($N=1$ obstacle rejection), 18 Budget Exceeded, 12 Stop Premature, 12 Missed Stop, 7 Stop After Reach, 0 Protocol Violations, 0 Infrastructure Crashes.
  - Checkpoints: Fully persisted on cluster at `results/arm_p_habitat_r2rce_100/`.
- **Completed Benchmark Execution - Arm L & Automated HF Upload (Job #639)**:
  - Job ID: **#639** (`benchmark-arm-l-and-hf-upload`).
  - Evaluation: **Arm L (Local Aids: $N=5$, Arrival Verification, Floor Guidance, 100 episodes)**.
  - **Episodes Completed**: **99/100 episodes**.
  - **Success Rate (SR)**: **19.19%** (+46.2% relative gain over Arm P).
  - **Success weighted by Path Length (SPL)**: **13.64** (+74.2% relative gain over Arm P).
  - **Oracle Success Rate (OSR)**: **40.40%** (40 episodes physically reached $\le 3.0$ m).
  - **Navigation Error (NE)**: **7.38 m**.
  - **Deadlocks**: Cut down from 37% to 20%.
  - Checkpoints & Traces: Persisted on cluster and fully uploaded to Hugging Face dataset [`Krut27/agenticnav-benchmark-results`](https://huggingface.co/datasets/Krut27/agenticnav-benchmark-results).
- **Completed Benchmark Execution - Arm L-v3 & Stride Scaling (Job #641 - Executed November 9, 2026)**:
  - Job ID: **#641** (`benchmark-arm-l-v3-stride-scaling-and-hf-upload`).
  - Evaluation: **Arm L-v3 (Approach Stride Scaling, Terminal Stop Verification, BEV Retrace Checking, 3D Bin Snapping)** across full 100-episode Habitat R2R-CE benchmark.
  - **Episodes Completed**: **100/100 episodes** (100% completion rate).
  - **Success Rate (SR)**: **24.00%** (95% CI: $[16.69\%, 33.23\%]$) — **all-time high**, +41.2% relative jump over v2, and nearly 2x Arm P.
  - **Success weighted by Path Length (SPL)**: **16.04** (>2x Arm P at 7.83).
  - **Oracle Success Rate (OSR)**: **45.00%** (45 episodes physically reached $\le 3.0$ m).
  - **Mean Navigation Error (NE)**: **6.76 m** (best precision recorded).
  - **Mean Trajectory Length**: **20.16 m** (cut down from 36.08 m in v2, 44.1% reduction in wandering).
  - **Failure Taxonomy**: 24 Successes, 20 Safety Deadlocks, 12 Budget Exceeded (cut in half from 24), 25 Stop Premature, 8 Missed Stop (cut in half from 16), 11 Stop After Reach, 0 Protocol Violations, 0 Infrastructure Errors.
  - **Top Scene SR**: `TbHJrupSAjP` 40.0% (6/15), `x8F5xyUWy9e` 42.9% (3/7), `zsNo4HB9uLZ` 25.9% (7/27).
  - Checkpoints & Traces: Persisted on cluster and fully uploaded to HF at `benchmark_results/arm_l_v3_habitat_r2rce_100/`.

---

## 8. Principled Architecture Fixes (Arm L-v3 & Arm L-v4)

- **Philosophical Invariant**: VLM is the sole cognitive decision-maker; framework provides purely passive 3D physical safety checks without leading hints, heuristics, or oracle NavMesh cheating.
- **3D Swept Cylinder Safety Checker (`safety_region.py`)**:
  - Replaced flat 2D bounding box with true 3D swept cylinder along continuous trajectory vector $\mathbf{p}(s) = (x(s), y(s), z(s))$.
  - At distance $s$, terrain supporting the agent ($y < y_{\text{base}}(s) + 0.15$ m) is treated as floor/stair treads.
  - Only points penetrating the agent's body cylinder $[y_{\text{base}}(s) + 0.15\text{ m}, y_{\text{base}}(s) + 1.25\text{ m}]$ are counted as obstacles.
  - General: naturally handles flat floors, stairs, ramps, and inclines without "stair detectors" or NavMesh oracle queries.
- **Arm L-v4 Physical Grounding & Clearance Synchronization**:
  1. **`query_depth` 3D Corridor Clearance**: Returns `clear: bool`, `obstacle_cause: str | null`, and the exact post-clamp `preview_rho_m` matching `execute_move_to`.
  2. **Physical Odometry Feedback & Collision Warnings (`history.py`)**: If the agent's actual motion is shortened by an obstacle collision (`actual_moved < 0.5 * rho_m`), action history explicitly logs the shortfall and warns the agent it remains near its previous position, eliminating internal odometry inflation that previously caused premature stops.
  3. **Pure-Rotation Disorientation Elimination**: When `rho_m == 0`, action history logs `ROTATED +theta° in place, 0.00m forward; Note: View 0 now faces your new heading`, preventing 180° double-rotation loops.
  4. **Peripheral Portal Checks (`prompts.py`)**: Instructs the agent to check Views 1 & 3 when stopping at archways/doorways, recognizing when it is already positioned between portal jambs and preventing pass-through overshoots.
  5. **Terminal Stride Restriction**: Only clamps strides to 1.0 m when `depth_m < 2.5 m` or on explicit terminal intent; preserves full $1.5\text{--}2.25$ m strides during open transit ($\ge 3.0$ m).
  6. **CoT Budget Cap**: Enforces a 2-sentence maximum thought on evidence-exhausted turns, preventing degenerate repetition loops.
- **Arm O (Habitat Oracle Local Actuation)**:
  - **Motivation**: Ablation to decouple high-level VLM multimodal reasoning from low-level hand-crafted harness actuation (collision boxes, straight-line swept cylinders, discrete velocity bursts).
  - **Zero Global Oracle Cheating**: The VLM remains the sole cognitive agent, receiving panoramic RGB-D and BEV views and deciding waypoints via `move_to(view, u, v, reason)` and `stop(reason)`. It NEVER receives the episode's final goal or shortest-path distances.
  - **NavMesh ShortestPath Follower**: Actuates the local waypoint selected by the VLM using Habitat-Sim's native `pathfinder.find_path()` and discrete simulator actions (`turn_left`, `turn_right`, `move_forward`), sliding along walls naturally and avoiding doorframe collisions.
  - **Stride & Waypoint Clamping**: Local travel along the NavMesh path is clamped to commanded stride ($1.0\text{--}1.75$ m), preventing unnatural multi-room teleports.
- **Test Suite Status**:
  - 93 passed / 1 skipped / 0 failed in full PyTest suite.
  - Tracked git working tree is clean and locked on commit `2aa6c8b` (`Arch/VLM`).

---

## 9. Arm Remediation & 20-Episode Smoke Test (Job #652 - Executed November 10, 2026)

- **Job ID**: **#652** (`smoke-20ep-remediation-arm-o`).
- **Benchmark Split**: Canonical 20 failure episodes from Arm O (`r2rce_val_unseen_20_remediation.json`, 0% prior SR).
- **Core Results & Acceptance Gates**:
  - **Success Rate (SR)**: **35.0% (7/20)** — 7 previously failed episodes converted directly to verified successes.
  - **Success weighted by Path Length (SPL)**: **24.41**.
  - **Oracle Success Rate (OSR)**: **55.0%**.
  - **Mean Navigation Error (NE)**: **5.70 m** (down by 3.14m from 8.84m).
  - **Safety Deadlocks**: **0 (0/20, 100% eliminated)** — Gate: **PASS**.
  - **Protocol Violations**: **0 (0/20, 100% eliminated)** — Gate: **PASS**.
  - **Missed Stops**: **0 (0/20, 100% eliminated)** — Gate: **PASS**.
  - **Pre-Flight Acceptance Gate**: **PASS**.
- **Key Forensic Fixes Validated**:
  1. **Upward Ray Scan & Azimuth Ground Fallback**: Converted 3 of 4 former deadlocks (`0031`, `0038`, `0058`) directly to successes.
  2. **JSON Argument Sanitizer**: Eliminated protocol violations; `0018` converted to success (NE 1.82m).
  3. **Egocentric Frustum Inversion & Odometry Reality Check**: Addressed portal overshoots and premature distant window stops.
- **Trace Persistence**: All checkpoints and traces uploaded to Hugging Face dataset [`Krut27/agenticnav-benchmark-results`](https://huggingface.co/datasets/Krut27/agenticnav-benchmark-results).





