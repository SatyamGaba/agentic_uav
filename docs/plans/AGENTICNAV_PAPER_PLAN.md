# AgenticNav: Zero-Shot Vision-and-Language Navigation — Master Implementation Specification & Replication Blueprint

**Reference Anchor Paper:** *AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness* (arXiv:2606.10577, 2026)  
**Authors:** Yijian Li, Changze Li, Han Zheng, Jiyuan Cai, Tong Qin (corresponding), Ming Yang (Shanghai Jiao Tong University and Yinwang Intelligent Technology)  
**Scope:** Complete, mathematically rigorous, self-contained replication blueprint for the R2R-CE benchmark (Habitat-Sim / Matterport3D) on the standard 100 `val_unseen` episodes with Gemini-3.7-Flash. Serves as a validated, frozen ground-robot baseline with modular interfaces for future aerial UAV adaptation.

---

## 1. System Architecture & Component Interaction

AgenticNav formulates Vision-and-Language Navigation (VLN) as an **active tool-calling sensory harness**. Rather than relying on black-box end-to-end neural policies or passive chat completions, the Vision-Language Model (VLM) operates as an embodied decision agent within a strictly firewalled execution loop.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                 VISION-LANGUAGE MODEL                                                  │
│                                                   (Gemini-3.7-Flash)                                                   │
└─────────────────────────────────┬──────────────────────────────────────────────────────▲───────────────────────────────┘
                                  │ JSON Function Calls                                  │ Tool Results, Recalled Views &
                                  │ (query_depth, recall, move_to, stop)                 │ Dynamic Observations
                                  ▼                                                      │
┌────────────────────────────────────────────────────────────────────────────────────────┴───────────────────────────────┐
│                                       AGENTICNAV HARNESS (PRIVILEGE FIREWALL)                                          │
│                                                                                                                        │
│   ┌─────────────────────────────────────────────────────────┐    ┌─────────────────────────────────────────────────┐   │
│   │                 PERCEPTION & SENSORY TOOLS              │    │                  EPISODIC MEMORY                │   │
│   │                                                         │    │                                                 │   │
│   │  query_depth(points)                                    │    │  ActionHistory                                  │   │
│   │  • Normalized (u, v) -> Integer pixel (x0, y0)          │    │  • Sliding window of last 6 executed steps      │   │
│   │  • Metric planar depth decoding                         │    │  • Stores (node transition, action, rationale)  │   │
│   │  • 5x5 window bilateral extraction                      │    │                                                 │   │
│   │  • Returns center depth + (min, mean, median) stats     │    │  EpisodicImageCache                             │   │
│   │                                                         │    │  • RGB buffers indexed by (node, view)          │   │
│   │  recall(node, view)                                     │    │  • Strict causality: 1 <= node <= t-1           │   │
│   │  • Multi-view RGB memory retrieval                      │    │                                                 │   │
│   │  • Dynamic turn-level observation augmentation          │    │  OccupancyGrid2D & BEVRenderer                  │   │
│   └─────────────────────────────────────────────────────────┘    │  • 256x256 log-odds allocentric map (0.15m/px)  │   │
│                                                                  │  • Trajectory polyline & numbered node markers  │   │
│   ┌─────────────────────────────────────────────────────────┐    │  • Red agent marker, heading vector & FOV wedge │   │
│   │                 KINEMATICS & SAFETY ENGINE              │    └─────────────────────────────────────────────────┘   │
│   │                                                         │                                                          │
│   │  move_to(view, u, v, reason)                            │    ┌─────────────────────────────────────────────────┐   │
│   │  • Pinhole inverse deprojection to optical frame        │    │                 MULTI-TURN HARNESS              │   │
│   │  • Extrinsic rotation T_k (yaw psi_k, mount height)     │    │                                                 │   │
│   │  • Range validation: finite planar z_c in [d_min, d_max]│    │  • Stateless per-step prompt reconstruction     │   │
│   │  • Standoff subtraction: rho = min(rho_max, D_raw - m)  │    │  • Evidence query budget: E_max = 4             │   │
│   │  • Floor quantization: rho_q = 0.25 * floor(rho / 0.25) │    │  • Consecutive reselect budget: R_max = 5       │   │
│   │  • Bearing quantization: theta_q = 15° * round(theta/15)│    │  • Malformed call budget: M_max = 3             │   │
│   │  • 3D pointcloud height slice [0.10m, 1.25m]            │    │  • Tool dispatch: evidence first, motion defer  │   │
│   │  • Corridor collision test: R(theta_q, rho_q + r_a)     │    │  • Deadlock & protocol violation termination    │   │
│   │                                                         │    └─────────────────────────────────────────────────┘   │
│   │  stop(reason)                                           │                                                          │
│   │  • Arrival declaration -> Episode termination           │                                                          │
│   └─────────────────────────────────────────────────────────┘                                                          │
└────────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                             │ Discrete Low-Level Primitives
                                                             │ (turn_left 15°, turn_right 15°, move_forward 0.25m)
                                                             ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                           SIMULATION ENVIRONMENT                                                       │
│                                (Habitat-Sim v0.2.4 / Matterport3D Dataset)                                             │
│                                                                                                                        │
│   • 4 Heading-Indexed Panoramic RGB Sensors (640x480, HFOV 90°, mount height 1.25m)                                    │
│   • 4 Metric Planar Depth Sensors (z_c in [0.5m, 5.0m], accessed strictly via tools)                                  │
│   • Odometry Pose Tracker (allocentric x, z, yaw)                                                                      │
│   • STRICT PRIVILEGE FIREWALL: Zero access to NavMesh PathFinder, top-down maps, or oracle goal coordinates during run  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Benchmark Reproduction Targets & Statistical Acceptance Protocol

### 2.1 Benchmark Results (R2R-CE `val_unseen`, 100 Episodes) `[P]`

The primary replication target is to match the published performance of Gemini-3.7-Flash on the 100 `val_unseen` episodes of R2R-CE (the exact subset used by Open-Nav, SmartWay, and EvoNav):

| Backbone Model | SR (%) $\uparrow$ | SPL (%) $\uparrow$ | OSR (%) $\uparrow$ | nDTW (%) $\uparrow$ | NE (m) $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gemini-3.7-Flash (Anchor Target)** | **76.0** | **66.50** | **78.0** | **69.38** | **3.48** |
| **Gemini-2.5-Pro** | 58.0 | 42.91 | 76.0 | 63.49 | 5.00 |
| **GPT-5.5** | 55.0 | 48.41 | 65.0 | 63.41 | 5.19 |
| **DeepSeek-V4-Flash Vision-Exp** | 49.0 | 36.48 | 64.0 | 61.18 | 5.67 |
| **Qwen3.8-27B** | 48.0 | 33.57 | 72.0 | 61.58 | 5.65 |

### 2.2 Published Ablation Hierarchy (Gemini-3.7-Flash) `[P]`

| Variant | Published SR (%) | Published SPL (%) | Margin vs Full (Δ Points) | Core Architectural Hypothesis Tested |
| :--- | :---: | :---: | :---: | :--- |
| **Full System (AgenticNav)** | **76.0** | **66.50** | — | Anchor baseline combining active tools, memory, and BEV |
| **w/o Action-Reasoning History** | 63.0 | 48.49 | $-13.0$ SR / $-18.01$ SPL | Sliding rationale queue prevents cyclical exploration |
| **Reasoning $\to$ Task-Progress Text** | 64.0 | 52.56 | $-12.0$ SR / $-13.94$ SPL | Self-generated rationales retain richer navigational state |
| **w/o BEV Image** | 70.0 | 59.46 | $-6.0$ SR / $-7.04$ SPL | Allocentric spatial awareness preserves global orientation |
| **Accumulate All Historical Images** | 66.0 | 54.05 | $-10.0$ SR / $-12.45$ SPL | Context cluttering degrades reasoning vs selective recall |
| **Action Tool $\to$ Waypoint Predictor** | 66.0 | 57.73 | $-10.0$ SR / $-8.77$ SPL | Click-based deprojection outperforms raw polar coordinates |
| **w/o Depth Tool** | 60.0 | 47.14 | $-16.0$ SR / $-19.36$ SPL | Active depth inspection resolves doorway & path clearance |
| **Depth Tool $\to$ Raw Depth Image** | 69.0 | 55.79 | $-7.0$ SR / $-10.71$ SPL | Symbolic metric queries yield higher precision than depth maps |
| **w/o Recall Tool** | 64.0 | 47.47 | $-12.0$ SR / $-19.03$ SPL | Visual memory verification enables recovery and backtracking |

### 2.3 Pre-Registered Decision Protocol `[I]`

To eliminate subjective evaluation, the replication outcome is governed by a formal statistical acceptance protocol:

1. **Repetitions**: Execute $K = 3$ repeated runs of the Full System on the 100 test episodes (ablation arms $K \ge 2$). Record random seeds, sampling hyperparameters ($\tau = 0.0$), and reasoning budgets.
2. **Episode Mean Record**: For each episode $i \in \{1 \dots 100\}$, calculate the mean success indicator $\overline{S}_i = \frac{1}{K} \sum_{k=1}^K S_{i,k}$ and mean SPL $\overline{\text{SPL}}_i = \frac{1}{K} \sum_{k=1}^K \text{SPL}_{i,k}$.
3. **Primary Consistency Criterion (Rule a)**:
   Compute non-parametric $95\%$ bootstrap Confidence Intervals ($B = 10,000$ resamples over the $N = 100$ episode mean records). The primary replication succeeds if and only if:
   $$76.0\% \in \text{CI}_{95\%}(\text{SR}) \quad \text{AND} \quad 66.50\% \in \text{CI}_{95\%}(\text{SPL})$$
4. **Paired Ablation Verification**:
   - For major ablations (**w/o Depth** and **w/o History**): One-sided paired McNemar test (for binary SR) and Wilcoxon signed-rank test (for continuous SPL) must demonstrate statistically significant superiority ($p < 0.05$, Holm-corrected).
   - For moderate ablations (w/o BEV, Depth $\to$ Depth Image, Reasoning $\to$ Task-Progress, w/o Recall, Accumulate All Images): Point estimates of the Full System must be strictly superior ($\Delta \text{SR} > 0, \Delta \text{SPL} > 0$).
5. **Reported Telemetry**: OSR, nDTW, NE, TL, tool calls per episode, Reselect counts and causes, deadlock rate, run-to-run standard deviation, and token expenditure.

---

## 3. Privilege Firewall & Simulation Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PRIVILEGE FIREWALL                              │
│                                                                        │
│   EXPOSED TO AGENT HARNESS:           STRICTLY FORBIDDEN TO HARNESS:   │
│   ✔ 4 Panoramic RGB Views             ✖ Habitat NavMesh PathFinder     │
│   ✔ 4 Depth Buffers (via tools only)  ✖ Pre-computed Top-Down Maps      │
│   ✔ Odometry Pose (x, z, yaw)         ✖ Geodesic Distance / Oracle Path│
│                                       ✖ Ground-Truth Goal Coordinates  │
│                                       ✖ Scene Semantic Mesh Labels     │
└────────────────────────────────────────────────────────────────────────┘
```

* **Simulator**: `habitat-sim` (v0.2.4 / v0.3.x pinned with `habitat-lab` VLN-CE configuration).
* **Dataset**: Matterport3D (MP3D) under the standard R2R-CE directory layout.
* **100-Episode Evaluation Set**: `data/episode_list_100.json` containing the exact episode IDs matching the published benchmark split.
* **Development Split**: 30 to 50 episodes for prompt and hyperparameter calibration, sourced from scenes **strictly absent from the test set** (val_seen or train splits).
* **Sensor Configuration**:
  - Horizontal FOV: $\text{HFOV} = 90^\circ$.
  - Resolution: $640 \times 480$ pixels.
  - Camera Mount: Height $h_{\text{cam}} = 1.25\,\text{m}$ above floor contact datum ($y=0$), zero pitch tilt.
  - Panoramic Views ($N=4$): View 0: Front ($0^\circ$), View 1: Right ($-90^\circ$), View 2: Back ($180^\circ$), View 3: Left ($+90^\circ$).
  - Depth Sensor Range: $[d_{\min}, d_{\max}] = [0.5\,\text{m}, 5.0\,\text{m}]$.
* **Action Budget**: $A_{\max} = 500$ low-level simulator discrete actions (`turn_left 15°`, `turn_right 15°`, `move_forward 0.25m`).
* **Success Criterion**: Geodesic distance to goal $\le 3.0\,\text{m}$ `[P]` **with `stop` called** `[I]`.

---

## 4. Coordinate Systems, Pinhole Geometry & Kinematics

### 4.1 Canonical Right-Handed Agent Frame
To prevent coordinate mirroring and sign errors, all geometric operations are performed in a unified right-handed Cartesian frame matching Habitat-Sim:
* **Forward**: $-Z_{\text{agent}}$
* **Right**: $+X_{\text{agent}}$
* **Up (Vertical)**: $+Y_{\text{agent}}$ (Floor contact datum $y=0$, camera center at $y = 1.25\,\text{m}$)
* **Heading / Yaw ($\theta$)**: Counter-Clockwise (CCW) positive around $+Y_{\text{agent}}$.

Camera optical frame: $+x$ right, $+y$ down, $+z$ forward (along optical axis).

### 4.2 Pinhole Deprojection & Extrinsic Transform

#### 1. Integer Pixel Conversion
Given normalized image coordinates $(u, v) \in [0.0, 1.0]$ in view $k$, convert to 0-indexed integer pixel coordinates using floor-clamping:
$$x_0 = \text{clip}\left(\lfloor u \cdot W \rfloor, \; 0, \; W - 1\right), \quad y_0 = \text{clip}\left(\lfloor v \cdot H \rfloor, \; 0, \; H - 1\right)$$

#### 2. Deprojection to Optical Frame
Using focal length $f_x = \frac{W}{2 \tan(\text{HFOV}/2)}$, $f_y = f_x$, and principal point $(c_x, c_y) = (W/2, H/2)$, the optical 3D vector $\mathbf{p}_c = [x_c, y_c, z_c]^T$ with half-pixel centering $(+0.5)$ is:
$$x_c = \frac{(x_0 + 0.5 - c_x) \cdot z_c}{f_x}, \quad y_c = \frac{(y_0 + 0.5 - c_y) \cdot f_y}{f_y}, \quad z_c = z_c$$

#### 3. View Extrinsic Rotation ($T_k$)
Compensate for camera height ($h_{\text{cam}} = 1.25\,\text{m}$) and rotate by CCW view azimuth $\psi_k \in \{0^\circ, -90^\circ, 180^\circ, +90^\circ\}$ around $+Y$:
$$\begin{bmatrix} p_x \\ p_y \\ p_z \end{bmatrix} = \begin{bmatrix} \cos\psi_k & 0 & \sin\psi_k \\ 0 & 1 & 0 \\ -\sin\psi_k & 0 & \cos\psi_k \end{bmatrix} \begin{bmatrix} x_c \\ h_{\text{cam}} - y_c \\ -z_c \end{bmatrix}$$

*Verification*:
* View 0 (Front, $\psi_0 = 0^\circ$): Optical ray straight ahead $(x_c=0, z_c=2.0)$ maps to $(0, 1.25, -2.0)$ (Forward $-Z$).
* View 1 (Right, $\psi_1 = -90^\circ$): Optical ray straight ahead maps to $(+2.0, 1.25, 0)$ (Right $+X$).
* View 2 (Back, $\psi_2 = 180^\circ$): Optical ray straight ahead maps to $(0, 1.25, +2.0)$ (Back $+Z$).
* View 3 (Left, $\psi_3 = +90^\circ$): Optical ray straight ahead maps to $(-2.0, 1.25, 0)$ (Left $-X$).

### 4.3 Bearing Angle & Corridor Vectors
In the canonical agent frame:
$$\theta = \text{atan2}(-p_x, \; -p_z), \quad \mathbf{b}_\theta = \begin{bmatrix} -\sin\theta \\ -\cos\theta \end{bmatrix}, \quad \boldsymbol{\ell}_\theta = \begin{bmatrix} \cos\theta \\ -\sin\theta \end{bmatrix}$$
* $\theta$ is positive to the left (CCW) and negative to the right (CW).
* $\mathbf{b}_\theta$ is the unit forward vector along bearing $\theta$.
* $\boldsymbol{\ell}_\theta$ is the unit lateral vector perpendicular to $\mathbf{b}_\theta$ (pointing to the right of the corridor).

### 4.4 Continuous Displacement & Discrete Quantization
1. **Horizontal Distance**: $D_{\text{raw}} = \sqrt{p_x^2 + p_z^2}$.
2. **Displacement Formula `[P]`**:
   $$\rho = \min\left(\rho_{\max}, \; \max\left(0.0, \; D_{\text{raw}} - m\right)\right)$$
   where $m = 0.30\,\text{m}$ (standoff margin) and $\rho_{\max} = 2.5\,\text{m}$. Any target closer than $m$ yields $\rho = 0.0$ and never a negative value.
3. **Quantization `[I]`**:
   $$\theta_q = 15^\circ \times \text{round}\left(\frac{\theta}{15^\circ}\right), \quad \rho_q = 0.25\,\text{m} \times \left\lfloor \frac{\rho}{0.25\,\text{m}} \right\rfloor$$
   Strict floor quantization ($\lfloor \dots \rfloor$) guarantees that discrete motion never rounds upward into obstacle boundaries.
4. **Zero-Displacement Policy (`zero_rho_policy`)**:
   - If $\rho_q = 0$ and $\theta_q \neq 0$: with `zero_rho_policy = rotate` (default), execute a **turn-only move** (`forward_steps = 0`, `turn_steps != 0`).
   - If both $\rho_q = 0$ and $\theta_q = 0$: reject with cause `no_motion`.

### 4.5 Safety Corridor Collision Check
* **Corridor Geometry $\mathcal{R}$**:
  $$\mathcal{R} = \left\{ s \cdot \mathbf{b}_{\theta_q} + h \cdot \boldsymbol{\ell}_{\theta_q} \;\Big|\; 0 < s < \rho_q + r_a, \quad |h| < r_a \right\}$$
  where $r_a = 0.18\,\text{m}$ is the robot chassis radius.
* **Obstacle Pointcloud $\mathcal{O}$**: Multi-view depth buffers are unprojected into the agent frame and filtered by chassis height slice $[y_{\min}, y_{\max}] = [0.10\,\text{m}, 1.25\,\text{m}]$ to reject floor contact terrain ($y < 0.10\,\text{m}$) and ceiling points.
* **Collision Check**: For each 2D ground point $\mathbf{q} = (q_x, q_z) \in \mathcal{O}$, compute longitudinal distance $s = \mathbf{q} \cdot \mathbf{b}_{\theta_q} = -q_x \sin\theta_q - q_z \cos\theta_q$ and lateral distance $h = \mathbf{q} \cdot \boldsymbol{\ell}_{\theta_q} = q_x \cos\theta_q - q_z \sin\theta_q$.
  $$\text{Collision} \iff \exists \mathbf{q} \in \mathcal{O} \text{ s.t. } 0 < s < \rho_q + r_a \quad \text{and} \quad |h| < r_a$$
  If colliding, `move_to` is rejected with cause `obstacle`.

---

## 5. Technical Implementation of Online BEV Mapping

The Bird's-Eye View (BEV) map $B_t$ provides the model with allocentric spatial awareness. It is synthesized dynamically online using only ego-motion odometry and active depth pointclouds, preserving the strict privilege firewall.

```
       Allocentric World Frame                                  Grid Pixel Coordinates
        (Start pose at origin)                                     (256 x 256 canvas)
       
              -Z (North)                                                 Row 0
                  ▲                                                        ▲
                  │                                                        │
                  │                                                        │
     -X (West) ───┼───► +X (East)                           Col 0 ─────────┼─────────► Col 255
                  │                                                        │ (Center at 128, 128)
                  │                                                        │
                  ▼                                                        ▼
              +Z (South)                                                Row 255
```

### 5.1 Grid Coordinate System & Transformations
* **Resolution**: $s = 0.15\,\text{m/pixel}$.
* **Canvas Size**: $G = 256 \times 256$ pixels.
* **Physical Coverage**: $G \cdot s = 38.4\,\text{m} \times 38.4\,\text{m}$ centered at the start pose $(x=0, z=0)$.
* **World-to-Grid Mapping**:
  $$\text{col} = \text{round}\left(\frac{G}{2} + \frac{x_{\text{world}}}{s}\right), \quad \text{row} = \text{round}\left(\frac{G}{2} + \frac{z_{\text{world}}}{s}\right)$$
* **Grid-to-World Mapping**:
  $$x_{\text{world}} = \left(\text{col} - \frac{G}{2}\right) \cdot s, \quad z_{\text{world}} = \left(\text{row} - \frac{G}{2}\right) \cdot s$$

### 5.2 Log-Odds Occupancy Grid Update Algorithm
The map maintains a continuous log-odds grid $L \in \mathbb{R}^{256 \times 256}$, initialized to $0.0$ (unknown space, $P = 0.5$).

1. **Pointcloud Transform to World Frame**:
   Let current agent odometry pose be $(x_t, z_t, \psi_t)$ where $\psi_t$ is CCW heading. For each obstacle point $(p_x, p_z)$ in the agent frame:
   $$x_{\text{world}} = x_t + p_x \cos\psi_t + p_z \sin\psi_t$$
   $$z_{\text{world}} = z_t - p_x \sin\psi_t + p_z \cos\psi_t$$
2. **Raycasting & Free Space Clearing**:
   For downsampled obstacle points, cast a ray from the agent grid cell $(c_{\text{agent}}, r_{\text{agent}})$ to the obstacle grid cell $(c_{\text{obs}}, r_{\text{obs}})$.
   Along the ray, intermediate cells prior to the obstacle are updated with the free log-odds increment:
   $$L(\text{row}, \text{col}) \leftarrow \max\left(-4.0, \; L(\text{row}, \text{col}) + l_{\text{free}}\right) \quad \text{with } l_{\text{free}} = \ln\frac{0.4}{0.6} \approx -0.405$$
3. **Obstacle Cell Update**:
   The terminal cell $(c_{\text{obs}}, r_{\text{obs}})$ is updated with the occupied log-odds increment:
   $$L(r_{\text{obs}}, c_{\text{obs}}) \leftarrow \min\left(+4.0, \; L(r_{\text{obs}}, c_{\text{obs}}) + l_{\text{occ}}\right) \quad \text{with } l_{\text{occ}} = \ln\frac{0.7}{0.3} \approx +0.847$$
4. **Grayscale Map Conversion**:
   Convert log-odds grid to uint8 image:
   $$\text{Pixel Value} = \begin{cases} 255 \text{ (Free space, White)} & \text{if } L < -0.4 \\ 0 \text{ (Occupied boundary, Black)} & \text{if } L > +0.4 \\ 127 \text{ (Unexplored / Unknown, Gray)} & \text{otherwise} \end{cases}$$

### 5.3 Canvas Rendering Pipeline & Visual Overlays
The uint8 grayscale map is expanded to a 3-channel RGB image ($256 \times 256 \times 3$) and rendered with distinct navigational markers:

1. **Trajectory Polyline**:
   Connect sequential decision node positions $(x_1, z_1) \to (x_2, z_2) \dots \to (x_t, z_t)$ with a solid cyan line (`RGB: (0, 150, 255)`, thickness: 2 px).
2. **Numbered Recall Node Badges ($1 \dots t-1$)**:
   For each past decision point $i \in \{1 \dots t-1\}$:
   - Draw an amber circle at $(c_i, r_i)$ with radius $r = 6\,\text{px}$ (`RGB: (255, 200, 0)`, black outline: 1 px).
   - Render the integer string `$i$` centered within the badge in black font. These numbers strictly match the `node` argument in `recall(node, view)`.
3. **Current Agent Marker & Heading Indicator (Node $t$)**:
   - Draw a solid red circle at current position $(c_t, r_t)$ with radius $r = 5\,\text{px}$ (`RGB: (255, 0, 0)`, white outline: 1 px).
   - Compute forward unit vector on canvas: $d_{\text{col}} = -\sin\psi_t, d_{\text{row}} = -\cos\psi_t$.
   - Draw a red heading vector arrow of length $14\,\text{px}$ from current position to tip $(c_t + 14 d_{\text{col}}, r_t + 14 d_{\text{row}})$.
4. **Front Camera Field-of-View Wedge**:
   To ground the front view (View 0), render two red rays from current position spanning $\pm 45^\circ$ ($\text{HFOV} = 90^\circ$) of length $22\,\text{px}$ (`RGB: (255, 50, 50)`, thickness: 1 px).

---

## 6. Technical Implementation of Tool Internals

### 6.1 `query_depth` Tool Internals
* **JSON Signature**:
  ```json
  {
    "name": "query_depth",
    "description": "Query metric depth at one or more normalized pixel coordinates across current camera views. Returns center depth and 5x5 spatial statistics (min, mean, median) in meters.",
    "parameters": {
      "type": "object",
      "properties": {
        "points": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "view": {"type": "integer", "enum": [0, 1, 2, 3]},
              "u": {"type": "number", "minimum": 0.0, "maximum": 1.0},
              "v": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["view", "u", "v"]
          }
        }
      },
      "required": ["points"]
    }
  }
  ```
* **Execution Pipeline**:
  1. **Argument Parsing & Pixel Mapping**: For each point in the batch, extract $(u, v)$ and compute $(x_0, y_0) = (\text{clip}(\lfloor u \cdot W \rfloor, 0, W-1), \text{clip}(\lfloor v \cdot H \rfloor, 0, H-1))$.
  2. **Metric Depth Decoding**: Read depth buffer value `raw_val` at $(y_0, x_0)$.
     - If `raw_val` is NaN, Inf, $\le 0.0$, or saturated ($\ge 1.0$ when normalized): return error `{"view": view, "u": u, "v": v, "error": "invalid_depth"}`.
     - Convert raw depth: $d_{\text{center}} = d_{\min} + \text{raw} \cdot (d_{\max} - d_{\min})$ if normalized, or $d_{\text{center}} = \text{raw\_val}$ if metric.
     - If finite $d_{\text{center}}$ lies outside $[d_{\min}, d_{\max}]$: return error `out_of_range`.
  3. **$5 \times 5$ Patch Extraction**: Extract subarray $y \in [\max(0, y_0 - 2), \min(H, y_0 + 3)]$, $x \in [\max(0, x_0 - 2), \min(W, x_0 + 3)]$.
  4. **Patch Statistics**: Decode all patch pixels to metric meters. Filter valid pixels satisfying $d_{\min} \le d \le d_{\max}$.
     - If valid pixel count $\ge 3$: calculate `min_m`, `mean_m`, `median_m`.
     - Return valid result:
       `{"view": view, "u": u, "v": v, "depth_m": round(d_center, 2), "min_m": round(min, 2), "mean_m": round(mean, 2), "median_m": round(med, 2)}`
     - If valid pixel count $< 3$: return sparse fallback:
       `{"view": view, "u": u, "v": v, "depth_m": round(d_center, 2), "min_m": null, "mean_m": null, "median_m": null, "sparse_patch": true}`

### 6.2 `recall` Tool Internals
* **JSON Signature**:
  ```json
  {
    "name": "recall",
    "description": "Retrieve a past RGB observation from a previously visited decision node (1 <= node <= t-1).",
    "parameters": {
      "type": "object",
      "properties": {
        "node": {"type": "integer"},
        "view": {"type": "integer", "enum": [0, 1, 2, 3]}
      },
      "required": ["node", "view"]
    }
  }
  ```
* **Execution Pipeline**:
  1. **Causality & Range Validation**:
     - Check $1 \le \text{node} \le t-1$. If $\text{node} \ge t$ or $\text{node} < 1$, reject with error message: `"Invalid node={node}. Recall is only permitted for past visited nodes (1 <= node <= t-1). Current node is {t}."`
     - Check $0 \le \text{view} \le 3$. If invalid, return error.
  2. **Image Retrieval & Memory Formatting**:
     - Fetch RGB array $C[\text{node}, \text{view}]$ from `EpisodicImageCache`.
     - Construct a tool response block containing the recalled image and metadata: `{"status": "success", "node": node, "view": view, "heading_deg": psi_view}`.
     - The recalled image is injected into the **active turn only** as an assistant observation; it does not persist across future steps.

### 6.3 `move_to` Tool Internals
* **JSON Signature**:
  ```json
  {
    "name": "move_to",
    "description": "Command the robot to navigate towards a visual target (u, v) in view. Checks safety corridor and quantizes motion.",
    "parameters": {
      "type": "object",
      "properties": {
        "view": {"type": "integer", "enum": [0, 1, 2, 3]},
        "u": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "v": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason": {"type": "string"}
      },
      "required": ["view", "u", "v", "reason"]
    }
  }
  ```
* **Execution Pipeline**:
  1. Read center depth $z_c$ at $(u, v)$ in `view`. If NaN, Inf, or 0 $\to$ return `{"status": "reselect", "cause": "invalid_depth"}`. If finite $z_c \notin [d_{\min}, d_{\max}] \to$ return `{"status": "reselect", "cause": "out_of_range"}`.
  2. Deproject to 3D point $\mathbf{p} = (p_x, p_y, p_z)$ in canonical agent frame via Section 4.2.
  3. Compute continuous kinematics: $D_{\text{raw}} = \sqrt{p_x^2 + p_z^2}$, $\theta = \text{atan2}(-p_x, -p_z)$, $\rho = \min(\rho_{\max}, \max(0, D_{\text{raw}} - m))$.
  4. Quantize: $\theta_q = 15^\circ \cdot \text{round}(\theta / 15^\circ)$, $\rho_q = 0.25\,\text{m} \cdot \lfloor \rho / 0.25\,\text{m} \rfloor$.
  5. If $\rho_q == 0$:
     - If $\theta_q \neq 0$: with `zero_rho_policy = rotate` (default), execute turn-only move.
     - If both $\rho_q = 0$ and $\theta_q = 0$: reject with `{"status": "reselect", "cause": "no_motion"}`.
  6. Execute corridor collision check on $\mathcal{R}(\theta_q, \rho_q + r_a)$.
     - If collision detected: return `{"status": "reselect", "cause": "obstacle"}`.
  7. If safe:
     - Return: `{"status": "executed", "theta_deg": theta_q, "rho_m": rho_q, "moved_m": rho_q, "collided": false}`.
     - Execute discrete actions in simulator: turn $N_{\text{turn}} = \text{round}(\theta_q / 15^\circ)$, step forward $N_{\text{fwd}} = \text{round}(\rho_q / 0.25\,\text{m})$.

### 6.4 `stop` Tool Internals
* **JSON Signature**:
  ```json
  {
    "name": "stop",
    "description": "Declare that the navigation goal has been reached and terminate the episode.",
    "parameters": {
      "type": "object",
      "properties": {
        "reason": {"type": "string"}
      },
      "required": ["reason"]
    }
  }
  ```
* **Execution Pipeline**:
  1. Record final agent position $(x_T, z_T, \psi_T)$.
  2. Append stop record to `ActionHistory`.
  3. Signal episode completion to the harness runner.
  4. Trigger privileged evaluation module to compute geodesic distance to goal and final benchmark metrics.

---

## 7. Multi-Turn Decision Loop & Guard Policy

### 7.1 Stateless Context Reconstruction
At every decision step $t$, the conversational context provided to the VLM is rebuilt completely from scratch:
1. **System Prompt**: Verbatim navigation instructions, sensor specifications, and tool descriptions.
2. **Goal Instruction**: The natural language instruction string $L$.
3. **Active Visual Observations**: The 4 current panoramic RGB images (Views 0, 1, 2, 3) and the active $256 \times 256$ BEV map image $B_t$.
4. **Action-Reasoning History ($H_t$)**: The last 6 executed steps. Earlier turn dialogues, transient probe calls, and rejected target attempts are **strictly discarded**.

### 7.2 Multi-Turn Dispatcher & Guard Budgets
* **Evidence Budget ($E_{\max} = 4$)**:
  - A batched `query_depth` counts as exactly 1 evidence call.
  - When $E_{\max}$ is reached, `query_depth` and `recall` are stripped from the active tool schema, and the harness explicitly instructs the model that `move_to` or `stop` is required.
* **Reselect Budget ($R_{\max} = 5$)**:
  - If the model proposes targets that trigger reselect 5 consecutive times within the same step, navigation terminates immediately as **Safety-Deadlock** (scored as a failure). No artificial creep fallback is executed.
* **Malformed Call Budget ($M_{\max} = 3$)**:
  - Replies with no tool call or invalid argument JSON return a structured error prompt and increment the malformed counter. Exceeding $M_{\max} = 3$ terminates the episode as **Protocol-Violation**.
* **Simultaneous Tool Resolution**:
  - If the model returns both an evidence tool (`query_depth` or `recall`) and a terminal tool (`move_to` or `stop`) in the same turn:
    1. Execute evidence calls in order.
    2. Answer terminal tool with `{"status": "deferred", "message": "Evidence query executed first. Please inspect evidence and reissue your move decision."}`.
    3. Re-prompt model to finalize move after reviewing evidence.
  - If multiple terminal tools are called without evidence, execute the first and answer remaining calls with `{"status": "ignored"}`.
  - **Every tool call strictly receives a response**.

---

## 8. Priority Waterfall Failure Taxonomy

Failed episodes are classified using a strict, non-overlapping priority waterfall:

```
                                  [ Episode Finished ]
                                            │
                                  Is Episode Successful?
                                 (Stop called & NE <= 3m)
                                      /          \
                                    YES           NO
                                    /              \
                               [ SUCCESS ]      Is Infrastructure Error?
                                                (API timeout / 5x fail)
                                                   /              \
                                                 YES               NO
                                                  /                 \
                                    [ INFRASTRUCTURE-FAIL ]     Is OSR == 0 & Stop Called?
                                    (Exclude & Re-run)             /              \
                                                                 YES               NO
                                                                 /                  \
                                                      [ STOP-PREMATURE ]         Is OSR == 1?
                                                                                /          \
                                                                              YES           NO
                                                                              /              \
                                                                   [ STOP-AFTER-REACH ]    Is R_max Reached?
                                                                                              /          \
                                                                                            YES           NO
                                                                                            /              \
                                                                                 [ SAFETY-DEADLOCK ]   Is M_max Exceeded?
                                                                                                          /          \
                                                                                                        YES           NO
                                                                                                        /              \
                                                                                            [ PROTOCOL-VIOLATION ]  Is A_max Reached?
                                                                                                                       /          \
                                                                                                                     YES           NO
                                                                                                                     /              \
                                                                                                           [ BUDGET-EXCEEDED ] [ UNCLASSIFIED ]
```

1. **Infrastructure Failure**: API timeout, rate limit, or socket error after 5 retries. Excluded from scoring and re-run.
2. **Stop-Premature**: $\text{OSR} = 0$ (never reached goal radius) and `stop` was called.
3. **Stop-After-Reach**: $\text{OSR} = 1$ (visited goal radius $\le 3\,\text{m}$), but failed to succeed (stopped outside $3\,\text{m}$, or never called `stop`).
4. **Safety-Deadlock**: $\text{OSR} = 0$ and $R_{\max} = 5$ consecutive reselects reached.
5. **Protocol-Violation**: $\text{OSR} = 0$ and $M_{\max} = 3$ malformed/no-tool calls exceeded.
6. **Budget-Exceeded**: $\text{OSR} = 0$ and $A_{\max} = 500$ low-level simulator actions exhausted.
7. **Unclassified**: All remaining failure edge cases.

---

## 9. Evaluation Metrics & Verification Protocol

### 9.1 Metric Mathematical Formulations
* **Navigation Error (NE)**: Final geodesic distance from agent stop pose to goal:
  $$\text{NE} = d_{\text{geodesic}}(\mathbf{x}_T, \; \mathbf{x}_{\text{goal}})$$
* **Success Rate (SR)**: Binary indicator requiring stop called within goal threshold:
  $$\text{SR} = \mathbb{I}\left(\text{NE} \le 3.0\,\text{m} \quad \text{and} \quad \text{called\_stop} = \text{True}\right)$$
* **Oracle Success Rate (OSR)**: Binary indicator of visiting goal region at any point:
  $$\text{OSR} = \mathbb{I}\left(\min_{t \in \{1 \dots T\}} d_{\text{geodesic}}(\mathbf{x}_t, \; \mathbf{x}_{\text{goal}}) \le 3.0\,\text{m}\right)$$
* **Trajectory Length (TL)**: Cumulative Euclidean distance traversed:
  $$\text{TL} = \sum_{t=1}^{T-1} \|\mathbf{x}_{t+1} - \mathbf{x}_t\|_2$$
* **Success weighted by Path Length (SPL)**:
  $$\text{SPL} = \frac{1}{N_{\text{ep}}} \sum_{i=1}^{N_{\text{ep}}} S_i \cdot \frac{L_i}{\max\left(L_i, \; \text{TL}_i\right)}$$
  where $L_i$ is oracle shortest path distance from start to goal.

---

## 10. Complete Parameter Reference Table

| Category | Parameter | Value | Tag | Source / Specification |
| :--- | :--- | :---: | :---: | :--- |
| **Benchmark** | Benchmark Split | R2R-CE `val_unseen` (100 episodes) | `[P]` | Standard Open-Nav / SmartWay split |
| | Evaluation Success Threshold | $d_{\text{success}} \le 3.0\,\text{m}$ | `[P]` | R2R-CE standard criterion |
| | Simulator Action Budget | $A_{\max} = 500$ discrete actions | `[P]` | VLN-CE standard budget |
| **Sensors** | Horizontal FOV | $\text{HFOV} = 90^\circ$ | `[P]` | Paper Section 3.1 |
| | Image Resolution | $640 \times 480$ px | `[P]` | Standard Habitat camera aspect ratio |
| | Camera Mount Height | $h_{\text{cam}} = 1.25\,\text{m}$ | `[P]` | Standard VLN-CE ground robot mount |
| | Camera Pitch Tilt | $0^\circ$ (horizontal) | `[P]` | Fixed forward horizon |
| | Panoramic Views ($N$) | 4 views ($0^\circ, -90^\circ, 180^\circ, +90^\circ$) | `[P]` | Heading-indexed panoramic rig |
| | Depth Sensor Range | $[0.5\,\text{m}, 5.0\,\text{m}]$ | `[P]` | Matterport3D sensor limits |
| **Kinematics** | Robot Chassis Radius | $r_a = 0.18\,\text{m}$ | `[P]` | Standard Habitat LoCoBot radius |
| | Base Standoff Margin | $m = 0.30\,\text{m}$ | `[?]` | Calibrated on disjoint dev split |
| | Max Linear Displacement | $\rho_{\max} = 2.5\,\text{m}$ | `[?]` | Calibrated on disjoint dev split |
| | Discrete Turn Increment | $15^\circ$ | `[P]` | Habitat-Sim discrete action space |
| | Discrete Forward Increment | $0.25\,\text{m}$ | `[P]` | Habitat-Sim discrete action space |
| | Obstacle Height Slice | $[0.10\,\text{m}, 1.25\,\text{m}]$ | `[I]` | Chassis clearance filter |
| | Standoff Mode | `fixed` | `[P]` | Fixed margin (slope mode is ablation) |
| | Adjacent Bin Snap | `false` | `[I]` | Snapping disabled by default |
| | Zero Rho Policy | `rotate` | `[P]` | Turn-only move if $\rho_q=0, \theta_q \neq 0$ |
| **Memory & BEV**| History Window Size | 6 steps | `[P]` | Paper Section 3.2 |
| | BEV Canvas Size | $256 \times 256$ px | `[P]` | Paper Section 3.2 |
| | BEV Resolution | $0.15\,\text{m/pixel}$ | `[P]` | Paper Section 3.2 |
| | Image Cache Capacity | All nodes of episode | `[P]` | Paper Section 3.2 |
| **Harness** | Max Evidence Queries | $E_{\max} = 4$ per step | `[P]` | Paper Section 3.3 |
| | Max Consecutive Reselects| $R_{\max} = 5$ | `[I]` | Deadlock prevention policy |
| | Max Malformed Calls | $M_{\max} = 3$ | `[I]` | Protocol violation policy |
| | Model Backbone | `gemini-3.7-flash` | `[P]` | Baseline anchor model |
| | Sampling Temperature | $\tau = 0.0$ | `[I]` | Deterministic reproduction |

*Legend: `[P]` = Explicitly stated in reference paper; `[I]` = Derived engineering specification; `[?]` = Tuned hyperparameter on disjoint development split.*

