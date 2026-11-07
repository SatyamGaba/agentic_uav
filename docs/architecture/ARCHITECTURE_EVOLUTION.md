# Architectural Evolution per Iteration
## Aerial Vision-and-Language Navigation (VLN)

This document tracks the structural and algorithmic progression of our Aerial VLN agents across all project iterations to maintain complete scientific transparency.

---

### Iteration 1: Discrete Policy Baseline
- **Paradigm**: Direct mapping from text narrative to discrete action codes.
- **Action Space**: OpenFly 10-class discrete commands (0–9).
- **Perception**: Single monocular RGB image per turn without spatial depth unprojection.
- **Limitations**:
  - Open-loop drift: errors compounded over multi-stride maneuvers without spatial memory.
  - Heading discretization deficit: cannot make fine orientation corrections between $\pm 30^\circ$ steps.

---

### Iteration 2: Map-Aware Navigation
- **Paradigm**: Integrated 2D Grid Mapping with instruction sub-clause parsing.
- **Core Components**:
  - Instruction Chunking Manager.
  - 2D Occupancy Grid accumulator.
  - Proximity stopping check.
- **Limitations**:
  - Flat 2D assumption caused frequent collisions with 3D multi-level urban structures (bridges, overhangs, variable rooftop elevations).

---

### Iteration 3: Cross-Modal Spatio-Temporal Core & BEV Mapping
- **Paradigm**: Attention-driven spatial navigation with active keyframing.
- **Core Components**:
  - `CrossModalSpatioTemporalAgent`: Cross-attention transformer fusing temporal action history, instruction embeddings, and multi-scale visual tokens.
  - `BEVMapper`: $64\text{m} \times 64\text{m}$ (0.5m resolution) log-odds Bayesian occupancy map.
  - `KeyframeTrigger`: Hybrid trigger firing on $12\text{m}$ Euclidean translation or 3 action steps.
  - `AirSimDepthProvider`: Real-time retrieval of `DepthPlanar` float32 sensor buffers from AirSim RPC.
- **Empirical Baseline**: Evaluated at $20.0\text{m}$ arrival threshold; achieved $\text{SR}=37.5\%$, $\text{OSR}=50.0\%$, $\text{NE}=26.82\text{m}$ on seen environments.

---

### Iteration 4: OmniNav Fast-Slow Grounded Continuous Representation
- **Paradigm**: Decoupled dual-system cognitive architecture (Kahneman System 1 / System 2).
- **Core Components**:
  1. **System 2 (Slow Deliberative Brain)**: High-level multimodal reasoning via Qwen3-VL on keyframes.
  2. **System 1 (Fast Reactive Reflex)**: High-rate continuous controller via Artificial Potential Fields (APF) with dynamic altitude envelopes.
  3. **2.5D Elevation Map**: Height-discriminating log-odds fusion (separates flyable ground obstacles like cars/trees from solid vertical building structures).
  4. **Open-Vocabulary Grounder (OWL-ViT)**: Pretrained zero-shot bounding box detector unprojected to 3D world coordinates ($w^*$).
  5. **Landmark Track Manager**: Multi-hypothesis 3D Kalman filter tracking landmark candidates with monotonic arrival confirmation gates.
- **Empirical Result**: Achieved strong collision avoidance in dense urban maps; arrival threshold configured at $18.0\text{m}$.

---

### Current Refined Framework: Unified Modular Closed-Loop Architecture
- **Paradigm**: Decoupled adapter-environment-backend pattern designed for journal-grade reproducibility and fair cross-baseline comparison.
- **Core Architectural Advancements**:
  1. **Dual-Threshold Metric Engine**: Evaluates both $20.0\text{m}$ (OpenFly standard) and $5.0\text{m}$ (strict precision) with 95% Wilson Confidence Intervals.
  2. **Corrected Kinematics Engine**: Replaces approximate arcs with complete 3D vector mid-heading integration $(\Delta x, \Delta y, \Delta z, \Delta\theta)$.
  3. **Dataset Orientation Alignment**: Directly extracts true starting yaw from `raw_yaws[0]` eliminating initial heading divergence.
  4. **Multi-Modal Visual Perception Pipeline**:
     - Remote Mode: Direct bridge to AirSim RPC (`ImageType.Scene` + `ImageType.DepthPlanar`).
     - Procedural Fallback: Calibrated 3D pinhole perspective renderer generating realistic geometric urban facades and exact metric depth maps.
  5. **Pure VLM Grounding**: All heuristic regex compilers and mock modules completely purged. Adapters strictly interface with live VLM token generation.

---

### Iteration 5: True End-to-End Closed-Loop Visual Servoing (Current Implementation)
- **Motivation**: In previous discrete runs, open-loop action queues compounded turn quantization errors ($\pm 30^\circ$), resulting in lateral cross-track drift ($\text{NE} > 50\text{m}$) and overshoot blind spots from single center-pixel depth queries.
- **Paradigm**: Pure reactive-deliberative closed-loop visual servoing executed at **every step**.
- **Core Architectural Advancements**:
  1. **Purged Open-Loop Action Queues**: `CalibratedNavAdapter` no longer pre-plans action sequences in `reset()`. Every single decision is generated dynamically per-step based on the fresh visual scene.
  2. **Per-Step Multimodal Perception**: At every time-step $t$, the drone receives:
     - Fresh rendered RGB frame (`observation.rgb_b64`)
     - Float32 metric depth map (`observation.depth_map`)
     - Full 3D pose (`observation.pose`)
  3. **Visual Grounding via Tool Calling**: Live `qwen3-vl:8b` model called at every turn using `navigate_step(u, v, step_distance_m, is_stop, reasoning)`:
     - $u \in [0.0, 1.0]$: Horizontal visual bearing to target landmark/corridor.
     - $v \in [0.0, 1.0]$: Vertical elevation ($0.0=\text{climb}$, $0.5=\text{level}$, $1.0=\text{descend}$).
     - $\text{step\_distance\_m} \in [2.0, 9.0]$: Forward step scale.
     - $\text{is\_stop}$: Visual arrival signal.
  4. **Multi-Patch Spatial Depth Probing (9-Point Grid)**:
     - Evaluates a 9-point spatial grid across camera FOV (center, left, right, top, bottom, and 4 quadrant corners).
     - Solves angled approach blindness: triggers verified stop when minimum frontal distance $\le 3.5\text{m}$ regardless of landmark eccentricity.
  5. **Continuous 2D-to-3D Geometric Deprojection**:
     - Metric depth $D(u^*, v^*)$ backprojected through pinhole camera intrinsics ($\mathbf{K}$):
       $$\Delta\theta = \text{atan2}\left(u^* - 0.5, \frac{f_x}{W}\right)$$
       $$dx = \rho \cos(\Delta\theta), \quad dy = \rho \sin(\Delta\theta), \quad dz = -(v^* - 0.5) \cdot 5.0$$
     - Dynamic collision shielding clamps $\rho = \min(\text{step\_distance\_m}, \max(1.5, D^* - 1.5))$.
     - Eliminates $30^\circ$ turn quantization drift entirely in continuous mode.
  6. **Grammatical Action-Boundary Chunking**:
     - Instruction parsing utilizes semantic boundary regex `r'[;.\n]|\b(?:then|next|finally|after that)\b|\band\s+(?=(?:fly|turn|head|proceed|advance|move|climb|descend|go)\b)'`.
     - Preserves compound adjectives (e.g. "green and brown building") without creating verbless fragments that trigger reasoning loops.
  7. **Dual-Kinematics CLI Support**:
     - Flag `--mode continuous` (default, exact 3D visual servoing) vs `--mode discrete` (calibrated OpenFly action code mapping).

---

### Iteration 6: Unified Dual-System Fast-Slow Cognitive Architecture (Journal Paper Target)
- **Reference**: Detailed formal specification in [ITERATION_6_FAST_SLOW_SPEC.md](file:///c:/Users/Kruti/Desktop/PROG/UAV/docs/architecture/ITERATION_6_FAST_SLOW_SPEC.md).
- **Paradigm**: Decoupled asynchronous dual-system architecture uniting high-rate continuous flight control ($20\text{--}50\text{ Hz}$) with high-level deliberative multimodal spatial reasoning ($0.1\text{--}0.5\text{ Hz}$).
- **Core Architectural Advancements**:
  1. **Deterministic Fast Continuous Controller (System 1)**:
     - 6-DoF continuous kinematic velocity profiling ($\mathbf{v}_{\text{cmd}}, \dot{\psi}$) with dynamic jerk and acceleration limiting.
     - **In-Loop Odometry-Depth Distance Regulator**: Couples dense metric depth maps directly with real-time UAV odometry $(\mathbf{p}_t, \mathbf{v}_t)$ to calculate exact dynamic stopping distance ($d_{\text{stop}} = v\cdot t_{\text{react}} + \frac{v^2}{2a_{\text{brake}}} + d_{\text{margin}}$), resolving VLM depth blindness and preventing terminal overshoot.
     - **3D Artificial Potential Field (APF)** with ray clamping and dynamic vertical envelope protection.
     - **Active 2-Axis Gimbal Pitch/Yaw Tracking**: Servos camera pitch ($\theta_g = -\text{atan2}(\Delta z, d_{\text{horiz}})$) and yaw to keep approach corridors and target centroids locked within the high-acuity central cone.
     - **High-Rate Visual Tracking**: Real-time CSRT/ByteTrack (or lightweight edge detectors like YOLO-World/NanoOWL) tracking targets at $30\text{ Hz}$ between VLM reasoning turns.
  2. **Deliberative Cognitive Backbone (System 2)**:
     - **Compulsory Feedback Invariant**: System 2 is a compulsory component of the closed feedback loop and cannot be bypassed.
     - Triggered dynamically via Euclidean progress ($\Delta d \ge 8\text{m}$), angular displacement ($|\Delta\psi| \ge 45^\circ$), waypoint arrival, scene graph mutation, or safety timeout ($5\text{s}$).
     - **Elimination of Rigid Phase State Machines**: Abolishes hardcoded `Search` $\to$ `Navigate` $\to$ `Maneuver` sequences and static camera profiles. High-level intent (`SEARCH_AND_ORIENT`, `CORRIDOR_TRANSIT`, `TARGET_APPROACH`, `SURVEY_INSPECT`, `TERMINAL_VERIFY`) is dynamically arbitrated by the VLM.
  3. **Zero-Shot 3D Metric Scene Graph & SLAM Anchoring**:
     - **Zero Hardcoded Volumetric Dictionaries**: Eliminates environment-fragile tables (`house: 6m`). Centroids and 3D bounding extents are computed directly via pinhole backprojection from sensor depth and/or SLAM map points ($P_c = D(u, v) \mathbf{K}^{-1} [u, v, 1]^T$).
     - **Spatial Deduplication & Anti-Hallucination**: Mahalanobis/Euclidean distance gating ($r < \max(2.5\text{m}, 0.5 \cdot \text{size})$) with Bayesian centroid fusion and multi-view observation gates ($N_{\text{obs}} \ge 2$).
     - **Structured Spatial Grounding**: Confirmed nodes are serialized into metric world-frame text prompts for the VLM.
     - **Monocular/Visual-LiDAR SLAM Interface**: 500 ORB keypoints/frame triangulated into an 8,000-point world map memory for accurate 3D bounding-box depth queries.
  4. **Target Memory Tracker & Removal of Coasting Mode**:
     - Anticipatory edge-boundary detection ($|u_t| > 0.7$) initiates centering yaw/pitch before targets leave FOV.
     - **Abolished Coasting Mode**: When target visual tracking is lost, the UAV executes a 3D-guided re-acquisition trajectory directly to the target's estimated global coordinates stored in the 3D Scene Graph while pointing the camera at the target.
     - **Budgeted Search Sweep**: If missing at the target coordinates, the drone executes a budgeted local scan ($\le 6$ sector turns) before requesting deliberate VLM re-planning.
  5. **Safety-Only Fallback State Machine**:
     - FSM logic is strictly demoted from mission progression to deterministic safety fallbacks (`NORMAL`, `EMERGENCY_REPULSION`, `3D_REACQUISITION`, `BUDGETED_SCAN`, `HOVER_AND_REPLAN`).
  6. **Decoupled Kinematics & Pluggable Benchmark Adapters**:
     - Full general-purpose 6-DoF continuous flight dynamics core.
     - External adapter layer (`OpenFlyBenchmarkAdapter`) encapsulates discrete action quantization and fixed-camera constraints without compromising the general framework.

