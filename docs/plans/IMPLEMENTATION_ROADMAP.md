# Scientific Implementation & Experimentation Roadmap
## Journal-Targeted Aerial VLN Project

---

### Phase 1: Local-Remote Integration & Script Synchronization (COMPLETED)
- [x] Establish bidirectional sync between local workspace and remote GPU server (`sync_framework_to_remote.py` and `sync_from_remote.py`).
- [x] Create comprehensive `RESEARCH_GUIDELINES.md` operational constitution.
- [x] Build living documentation hierarchy (`docs/architecture/`, `docs/results/`, `docs/failures/`, `docs/plans/`).
- [x] Publish comprehensive cross-iteration comparative study ([`docs/architecture/COMPARATIVE_FRAMEWORK_STUDY.md`](file:///c:/Users/Kruti/Desktop/PROG/UAV/docs/architecture/COMPARATIVE_FRAMEWORK_STUDY.md)).
- [x] Archive all existing remote logs and benchmark results to local `remote_backup/`.

---

### Phase 2: End-to-End Closed-Loop Architecture & Remote Benchmark (ACTIVE)
- [x] **Purge Open-Loop Queuing**: Replaced discrete trajectory pre-compilation with per-step visual servoing in `CalibratedNavAdapter`.
- [x] **Continuous 2D-to-3D Deprojection**: Integrated pinhole backprojection $(\Delta\theta = \text{atan2}(u^* - 0.5, f_x / W))$ with depth-based collision clamping.
- [x] **Multi-Patch Depth Probing**: Deployed 9-point spatial FOV grid to eliminate angled landmark arrival blindness ($d \le 3.5\text{m}$).
- [x] **Semantic Action-Boundary Chunking**: Implemented verb-lookahead boundary regex to prevent compound adjective fragmentation.
- [x] **Resilient Runner Execution**: Added exception trapping in `runner.py` for token-budget overflow and individual episode faults.
- [ ] Deploy refined framework to remote NVIDIA L40S server (`/home/jovyan/scratch/framework`).
- [ ] Connect `CalibratedNavAdapter` to remote live `qwen3-vl:8b` running on Ollama (`http://127.0.0.1:11434`).
- [ ] Await user confirmation before initiating the 40-episode benchmark.
- [ ] Execute 40-episode continuous visual servoing benchmark on Seen split (`benchmark/seen.json`).
- [ ] Automatically sync all resulting logs, JSON summaries, and trajectory artifacts to local `benchmark/` and `docs/results/`.

---

### Phase 3: Architectural Refinements for SOTA Performance
- [ ] **OmniNav Kalman Grounding Integration**: Bring the proven 3D Kalman landmark tracking from Iteration 4 into the refined framework adapter suite.
- [ ] **Dual-Gated Proximity Arrival**: Eliminate the $10\%$ OSR-to-SR overshoot gap by locking terminal `STOP` execution to metric depth thresholds ($d \le 3.5\text{m}$).
- [ ] **2.5D Elevation-Aware Collision Avoidance**: Integrate log-odds elevation mapping to prevent rooftop clamping.

---

### Phase 4: Full Benchmark Suite & Journal Paper Artifacts
- [ ] Complete full-scale evaluation across all 1800 seen and 1200 unseen OpenFly benchmark episodes on the remote L40S server.
- [ ] Generate publication-grade comparative performance tables with Wilson 95% CIs.
- [ ] Generate high-resolution trajectory overlay figures comparing agent flight paths against ground truth corridors.
