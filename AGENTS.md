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
- [ ] **H1–H3 (Cluster Dry Run)**: 3-episode non-locked dry run on remote GPU cluster, verifying manifest hashes, GPU EGL rendering, and checkpoint resume.
