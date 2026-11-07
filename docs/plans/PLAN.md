# AgenticNav: Architecture Framework & Replication Plan

**Anchor Paper:** *AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness* (arXiv:2606.10577v3, 2026)  
**Target Benchmark:** R2R-CE (Habitat-Sim / Matterport3D) • Standard 100 `val_unseen` episodes  
**Primary VLM Backbone:** Gemini-3.7-Flash (Zero-Shot Function Calling)

---

### 1. Goal and Scope

We replicate the core AgenticNav tool-calling framework on the continuous vision-and-language navigation benchmark R2R-CE using Habitat and Matterport3D. The primary objective is to **faithfully replicate the architecture, mechanisms, and decision-step lifecycle** of the active tool-calling harness—verifying that active depth inspection, visual recall, an allocentric compact map, short-term reasoning history, and geometric corridor safety reproduce coherent, collision-free embodied navigation without task-specific training.

Rather than fixating solely on matching exact published benchmark numbers as the only acceptance criterion, success is defined as:
1. **Fidelity of Mechanisms**: Faithful implementation of the paper's action grounding, safety corridor, depth statistics, visual memory, and multi-turn step loop.
2. **Verified Geometry & Lifecycle**: Zero leakage across the privilege firewall, mathematically verified corridor checks ($|h| < r_a$), and robust handling of all step transitions and edge cases.
3. **Reproducible Baseline & Telemetry**: Completing reproducible zero-shot benchmark runs on R2R-CE to establish an empirical reference and diagnostic telemetry for later adaptation to continuous 6-DoF aerial vehicles (UAVs) and new datasets.

---

### 2. Architecture and Responsibility Boundaries

The system is structured into five decoupled components with strict responsibility boundaries:

```
+----------------------------------------------------------------------------------------------------+
|                                         PRIVILEGE FIREWALL                                         |
|  PERMITTED (Agent Context): Multi-View RGB, Tool-Queried Depth, Egocentric Relative Odometry       |
|  QUARANTINED (Evaluator Only): NavMesh, Shortest Paths, Oracle Maps, Goal Coordinates (x_g,y_g,z_g) |
+----------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+────────────────────────────────────────────────────────────────────────────────────────────────────+
| ENVIRONMENT ADAPTER (Habitat-Sim / Future UAV Platforms)                                           |
|   - Provides Common Observation Contract: Synchronized RGBs, aligned metric depth, camera poses   |
|   - Owns Platform Controls: Habitat translates continuous motion into discrete primitives (15°,    |
|     0.25m) and returns actual executed displacement and collision signals.                         |
+───────────────────────────────────────────────────▲────────────────────────────────────────────────+
                                                    │ Motion Execution Request (Δθ, ρ)
                                                    │ Actual Movement Outcome
+───────────────────────────────────────────────────┴────────────────────────────────────────────────+
| ACTIVE HARNESS (Central Coordinator)                                                               |
|   - Context Assembly: Rebuilds stateless prompt (Instruction + Map + Views + Last 6 History)       |
|   - Memory Management: Allocentric BEV occupancy grid, episodic image cache, sliding history       |
|   - Tool Dispatcher & Step Lifecycle: Enforces evidence gathering, reselects, and guard budgets    |
+──────────────────────────▲────────────────────────────────────────▲────────────────────────────────+
                           │                                        │
           Pixel Queries   │ Tool Results            Candidate Target│ Safe Vector (Δθ, ρ)
           (Depth, Recall) │ (Stats, Cached Frame)      (view, u, v)│ or Reselect Feedback
                           │                                        │
+──────────────────────────▼────────────────────────+  +────────────▼────────────────────────────────+
| VLM ADAPTER (Gemini-3.7-Flash / Open Backbones)   |  | ACTION GROUNDING & SAFETY ENGINE            |
|   - Translates tool schemas into provider JSON    |  |   - Pinhole 3D Deprojection: (u,v,z) -> 3D  |
|   - Executes multimodal function calling          |  |   - Standoff & Cap: ρ = min(ρ_max, D - m)   |
|   - Stateless per-step prompt handling            |  |   - Corridor Check: |h| < r_a, 0 < s < ρ+r_a|
+───────────────────────────────────────────────────+  +─────────────────────────────────────────────+
```

#### Component Responsibilities

1. **Environment Adapter [Ours]**:
   - Isolates all simulator-specific and physical-platform assumptions.
   - Normalizes raw environment buffers into the **Common Observation Contract** (Section 3).
   - Translates continuous grounded motion $(\Delta\theta, \rho)$ into the platform's native actuation. For Habitat-Sim, it executes discrete primitives (`turn_left 15°`, `turn_right 15°`, `move_forward 0.25m`) and returns the *actual executed displacement* after physical collision checks.
2. **Active Harness [Paper, Ours]**:
   - The central coordinator. It owns the decision-step lifecycle, prompt formulation, memory updates, and tool dispatching.
   - Maintains zero simulator-specific dependencies, ensuring 100% portability between ground simulation and UAV flight.
3. **Action Grounding & Safety Engine [Paper]**:
   - Unprojects candidate pixel coordinates $(u, v)$ to 3D metric targets using pinhole geometry.
   - Evaluates the canonical rectangular safety corridor against body-height obstacle points before any motion is issued.
   - Returns either a validated motion command $(\Delta\theta, \rho)$ or structured `reselect` feedback.
4. **VLM Adapter [Ours]**:
   - Provider-specific bridge (Gemini-3.7-Flash) translating tool definitions into native function-calling formats.
   - The harness owns tool semantics; the adapter owns API protocol serialization.
5. **Evaluator [Ours]**:
   - Completely external to agent runtime. Evaluates episode paths against ground-truth shortest paths and goal coordinates exclusively after the agent emits `stop` or terminates.

---

### 3. Common Observation Contract & Lifecycle

#### 3.1 Observation Contract (Per Decision Step $t$)
At every decision node, the Environment Adapter delivers a standardized observation packet:
* **Multi-View RGB Images**: $N$ calibrated perspective frames ($N=4$ default: $0^\circ, -90^\circ, 180^\circ, +90^\circ$; or $N=1$ forward), $640 \times 480$, $90^\circ$ HFOV.
* **Aligned Metric Depth Buffers**: $640 \times 480$ float arrays representing true orthogonal planar distance in meters. Non-positive, NaN, or Inf values are flagged as `invalid_depth`. Values outside $[0.5, 5.0]\,\text{m}$ or saturated raw values are masked as `out_of_range`.
* **Camera Calibration**: Focal lengths $(f_x, f_y)$, principal points $(c_x, c_y)$, and extrinsic mounting offsets.
* **Agent Pose & Odometry**: Relative position $(x_t, y_t, \theta_t)$ in the canonical right-handed frame ($+X$ right, $+Y$ up, $-Z$ forward; $\theta$ positive CCW).

#### 3.2 Decision-Step Lifecycle
1. **Observation & Memory Update**:
   - New sensor observation is registered.
   - Depth pointclouds and pose update the allocentric 2D occupancy grid ($B_t$).
   - Current RGB views are stored in the episodic image cache under integer node index $t$.
2. **Stateless Context Rebuild**:
   - Harness generates a fresh multimodal prompt: Task Instruction + Rendered Map ($B_t$) + Current Camera Views + Last 6 Action History Records.
   - Active tools exposed: `query_depth`, `recall`, `move_to`, `stop`.
3. **Multi-Turn Interaction**:
   - **Evidence Query (`query_depth`, `recall`)**: The harness executes the query against the *current static observation* and returns metric stats or cached images. The agent does not move, and the simulator state is untouched.
   - **Candidate Move (`move_to`) - Blocked**: If the target corridor contains obstacle points, the harness returns `reselect` with the obstacle cause and distance. The VLM reconsiders its target within the same decision step.
   - **Candidate Move (`move_to`) - Safe**: The motion command is passed to the Environment Adapter, physical movement executes, the actual traversed distance is recorded into action history, and step $t$ cleanly terminates.
   - **Stop (`stop`)**: The agent declares arrival, the step loop halts immediately, and the episode is passed to the Evaluator.
4. **Boundary Invariants**:
   - Map, episodic image cache, and action history reset completely at episode start.
   - If the VLM emits both evidence and movement calls in one turn, evidence executes first and movement is marked `deferred`.
   - Unavailable recall ($p \ge t$ or non-existent node) returns a structured error and never hallucinates.

---

### 4. Canonical Action Grounding & Safety Corridor

The spatial safety engine bridges 2D pixel selections into verified physical motion:

#### 4.1 Pinhole Deprojection & Standoff [Paper]
1. Given normalized image coordinates $(u, v)$ in view $k$ and center depth $z_c$:
   $$x_c = (u \cdot W - c_x)\frac{z_c}{f_x}, \quad y_c = (v \cdot H - c_y)\frac{z_c}{f_y}, \quad z_c = z_c$$
2. Rotated by view azimuth $\psi_k \in \{0^\circ, -90^\circ, 180^\circ, +90^\circ\}$ into the robot frame:
   $$D = \sqrt{x_c^2 + z_c^2}, \quad \Delta\theta = \text{atan2}(-x_c, -z_c) + \psi_k$$
3. Target distance is bounded using the canonical standoff and maximum distance cap:
   $$\rho = \min(\rho_{\max}, \max(0, D - m))$$
   where $m = 0.30\,\text{m}$ is the standoff margin and $\rho_{\max} = 2.5\,\text{m}$ is the maximum step distance.

#### 4.2 Canonical Safety Corridor [Paper, §III-C, Eqs. 5–7]
The obstacle set $P_{\text{obs}}$ is formed by depth points falling within the robot's vertical chassis envelope:
$$P_{\text{obs}} = \{q_j \mid y_{\min} \le q_{j, y} \le y_{\max}\}$$
where $y_{\min} = 0.10\,\text{m}$ (excluding navigable floor) and $y_{\max} = 1.25\,\text{m}$ (excluding ceiling overhangs).

The safety corridor $R$ along heading unit vector $\mathbf{b}_\theta$ and lateral unit vector $\mathbf{l}_\theta$ is defined as:
$$R = \{ s \cdot \mathbf{b}_\theta + h \cdot \mathbf{l}_\theta \mid 0 < s < \rho + r_a, \quad |h| < r_a \}$$
* **Width**: Exactly $2r_a = 0.36\,\text{m}$ (with robot radius $r_a = 0.18\,\text{m}$). *(Note: We strictly use $2r_a$, rejecting wider $2(r_a+m)$ formulations that artificially block doorways).*
* **Length**: Extends from the agent center $s = 0$ forward to $\rho + r_a$.
* **Corridor Check**: If any point in $P_{\text{obs}}$ falls inside $R$, the move is rejected with `reselect`; otherwise, it is safe to execute.

---

### 5. Parameter Register: Paper Specifications vs. Implementation Choices

To maintain complete transparency and avoid mislabeling implementation defaults as paper facts, all parameters are cataloged below:

| Parameter | Value | Attribution | Purpose & Expected Effect |
| :--- | :---: | :---: | :--- |
| **Tool Paradigm** | 4 tools (`query_depth`, `recall`, `move_to`, `stop`) | **[Paper]** | Replaces rigid waypoint predictors with active sensory inspection. |
| **Depth Neighborhood** | $5 \times 5$ window (min, mean, median) | **[Paper]** | Noise-resilient metric depth sampling; min detects thin doorframes. |
| **History Window** | Last 6 (reason, action) records | **[Paper]** | Short-term task continuity without prompt bloat. |
| **Visual Recall** | Bounded cache, indexed by $(p, k)$ | **[Paper]** | On-demand historical image inspection; excluded from default prompt. |
| **Corridor Width** | $2r_a = 0.36\,\text{m}$ ($|h| < r_a$) | **[Paper]** | Matches physical chassis width; allows traversing standard doorways. |
| **Corridor Length** | $s \in (0, \rho + r_a)$ | **[Paper]** | Sweeps bounding box from origin to target stopping point. |
| **Robot Radius ($r_a$)** | $0.18\,\text{m}$ | **[Ours]** | Standard Matterport3D / LoCoBot chassis radius. |
| **Standoff Margin ($m$)** | $0.30\,\text{m}$ | **[Ours]** | Prevents agent from ramming into obstacle surface. |
| **Max Step Distance ($\rho_{\max}$)** | $2.5\,\text{m}$ | **[Ours]** | Upper bound on single-step movement to prevent overshoot. |
| **Chassis Height Envelope** | $[0.10, 1.25]\,\text{m}$ | **[Ours]** | Isolates structural obstacles from navigable floor and ceilings. |
| **Discrete Primitives** | Turn $15^\circ$, step $0.25\,\text{m}$ | **[Ours]** | Habitat-Sim discrete R2R-CE benchmark control interface. |
| **Max Low-Level Steps** | 500 primitives | **[Ours]** | Benchmark step ceiling pinned from baseline evaluation configs. |
| **BEV Grid Canvas** | $256 \times 256$ px @ $0.10\,\text{m/px}$ | **[Ours]** | Allocentric spatial coverage ($25.6\,\text{m}$ span) upscaled $2\times$. |
| **BEV Log-Odds** | $l_{\text{occ}} = +0.847, l_{\text{free}} = -0.405$ | **[Ours]** | Bayesian occupancy updates; floor & max-range rays clear space. |
| **Guard Budgets** | $E_{\max}=4, R_{\max}=5, M_{\max}=3$ | **[Ours]** | Prevents infinite API billing loops and catches deadlocks. |

---

### 6. Verification and Evaluation Protocol

#### 6.1 Benchmark Dataset
* **Dataset**: R2R-CE `val_unseen` 100-episode benchmark subset.
* **Provenance**: Pinned to the official Open-Nav benchmark split (`data/episode_list_100.json`).
* **Evaluation Granularity**: Trajectory Length (TL), Oracle Success Rate (OSR), and SPL are measured across **all low-level primitive steps**, matching official Habitat-Sim benchmark standards.

#### 6.2 Success Criteria
The implementation is evaluated on:
1. **Architectural & Geometric Verification**:
   - 100% pass rate on geometric deprojection, corridor collision tests ($|h| < r_a$), and step lifecycle state machine unit tests.
   - Strict privilege firewall audit: zero leakage of NavMesh, oracle distances, or goal coordinates.
2. **Reproducible Navigation Telemetry**:
   - 3 repeated benchmark runs ($K=3$) using Gemini-3.7-Flash with greedy sampling ($\tau = 0.0$).
   - Bootstrap 95% confidence intervals ($B = 10,000$ resamples) computed on per-episode mean scores.
   - Reporting full telemetry: SR, SPL, OSR, nDTW, NE, TL, tool call frequency per episode, collision rejection rate, and token expenditure.
   - Published numbers (76.0% SR, 66.50% SPL) serve as reference diagnostic targets rather than a pass/fail blocker for the framework.

---

### 7. Clarified Boundary for Later UAV Extension

The architectural separation established in this framework guarantees clean future extension to continuous 6-DoF aerial vehicles:
* **Preservation of 3D Coordinates**: The Action Grounding engine preserves the **full unprojected 3D target $(x, y, z)$** before applying 2D planar projection.
* **Swappable Flight Policy**:
  - For ground robots, the 2D planar projection and vertical chassis band $[y_{\min}, y_{\max}]$ are evaluated against the 2D corridor.
  - For aerial drones, a continuous UAV Environment Adapter takes over, replacing the 2D corridor with a 3D swept bounding cylinder/ellipsoid along the 3D flight trajectory.
* **Invariant Cognitive Harness**: The VLM prompt format, tool schemas (`query_depth`, `recall`, `move_to`, `stop`), and multi-turn decision loop remain 100% identical between ground simulation and 6-DoF continuous flight.
