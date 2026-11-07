# Iteration 6: Dual-System Fast-Slow Cognitive Architecture for Aerial Vision-and-Language Navigation (VLN)
## Architectural Specification & Formal Scientific Blueprint


---

## 1. Executive Summary & Design Philosophy

Aerial Vision-and-Language Navigation (Aerial VLN) requires an autonomous Unmanned Aerial Vehicle (UAV) to interpret complex, high-level natural language instructions (e.g., *"Fly between the two red brick manors, pass the wind turbine on your left, and inspect the solar panel on the low flat roof"*) and navigate safely through unmapped, visually cluttered 3D urban environments.

Prior iterations exposed fundamental tensions between high-level reasoning and physical flight dynamics:
1. **Discrete Action Quantization Deficit (Iteration 1 & OpenFly Baseline)**: Enforcing discrete $30^\circ$ yaw turns and fixed step displacements causes irreversible heading discretization errors ($\pm 15^\circ$), compounding into $>25\text{--}40\text{m}$ lateral cross-track drift and severe terminal overshoot.
2. **Volumetric Hallucinations & Lack of Generalizability (Iteration 4)**: Relying on hardcoded prior volume dictionaries (`OBJECT_3D_VOLUMES = {"house": (10, 8, 6), "car": (4.5, 2, 1.5)}`) catastrophically fails in zero-shot transfer to unseen cities where building heights, facade dimensions, and architectural geometries vary continuously.
3. **Rigid Mission Phase State Machines**: Hardcoding rigid sequential phases (`SearchPhase` $\to$ `NavigatePhase` $\to$ `ManeuverPhase`) with hardcoded yaw search sectors ($60^\circ$) and heuristic camera profiles (`{"house": pitch=-15°}`) strips the agent of semantic adaptability and fails when missions require fluid re-searching, branching, or looping.
4. **Blind Coasting Failure**: In Iteration 4, when a tracking target exited camera boundaries, the drone entered a 3-second "coasting mode" locking translations and blindly rotating in place, resulting in unanchored drift or lost targets.
5. **VLM Spatial & Metric Depth Blindness**: Frontier Vision-Language Models (VLMs) excel at semantic reasoning and zero-shot object identification but exhibit poor perception of metric depth, absolute physical distance, and spatial proximity from 2D monocular images.

### Iteration 6 Architectural Paradigm
Iteration 6 resolves these limitations by establishing a **Decoupled Dual-System (Fast-Slow) Cognitive Architecture**:
- **System 1 (Fast Continuous Loop, $20\text{--}50\text{ Hz}$)**: A deterministic, high-rate flight controller executing obstacle avoidance via 3D Artificial Potential Fields (APF), in-loop depth-odometry distance regulation, active 2-axis gimbal tracking, and high-rate target visual tracking (optionally aided by lightweight zero-shot models).
- **System 2 (Slow Reasoning Loop, $0.1\text{--}0.5\text{ Hz}$)**: The deliberative cognitive backbone. Decomposes instructions, builds and maintains an environment-agnostic **3D Metric Scene Graph** from RGB and depth metrics via pinhole backprojection and SLAM spatial anchoring (zero hardcoded volumetric dictionaries), deduplicates landmarks, filters hallucinations, and arbitrates open-ended intent without rigid FSM constraints.
- **Compulsory Feedback Invariant**: System 2 is a compulsory component of the feedback loop; it can never be bypassed. Fast loops execute guided subgoals, reporting continuous progress and spatial changes back to System 2.
- **Metric Target Re-Acquisition**: Blind coasting is eliminated. When a target is occluded or exits the FOV, the UAV executes a 3D-guided re-acquisition trajectory directly to the estimated global coordinates stored in the 3D Scene Graph, backed by a budgeted local scan.
- **General-Purpose 6-DoF UAV Core**: The architecture is fundamentally formulated for continuous 6-DoF aerial dynamics. Benchmark constraints (e.g. OpenFly fixed camera and discrete actions) are handled strictly via external test adapters.

```
+=============================================================================================================+
|                                    ITERATION 6 COGNITIVE ARCHITECTURE                                       |
+=============================================================================================================+
|                                                                                                             |
|   +-----------------------------------------------------------------------------------------------------+   |
|   |                       SYSTEM 2: SLOW DELIBERATIVE REASONING LOOP (0.1 - 0.5 Hz)                    |   |
|   |                                                                                                     |   |
|   |  Natural Language Instruction + Goal Context                                                        |   |
|   |           |                                                                                         |   |
|   |           v                                                                                         |   |
|   |  +-----------------------------+     +-------------------------------+     +---------------------+  |   |
|   |  | Open-Ended Task Decomposer  | --> | Multimodal Spatial Grounder   | --> | VLM Decision Engine |  |   |
|   |  | (No Rigid State Machine)    |     | (Pinhole + SLAM BBox Queries) |     | (Goal & Intent Arb) |  |   |
|   |  +-----------------------------+     +-------------------------------+     +---------------------+  |   |
|   |                 ^                                    ^                                |             |   |
|   |                 |                                    |                                v             |   |
|   |  +-----------------------------------------------------------------------+   +-------------------+  |   |
|   |  |                  GLOBAL 3D METRIC SCENE GRAPH                         |   | Structured Subgoal|  |   |
|   |  |  - Zero Hardcoded Dictionaries (Direct Point Cloud & Depth Extents)  |   | & Caution Envelope|  |   |
|   |  |  - Spatial Deduplication (r < 2.0m Gating) & Confidence Filtering    |   +-------------------+  |   |
|   |  |  - Monocular/Visual-LiDAR SLAM Point Memory (8,000 World Points)      |             |            |   |
|   |  +-----------------------------------------------------------------------+             |            |   |
|   +----------------------------------------------------------------------------------------|------------+   |
|                                                     ^ Compulsory Closed Loop Telemetry     | Subgoal/Path   |
|                                                     | & Metric Keyframe Triggers           v Directives     |
|   +-----------------------------------------------------------------------------------------------------+   |
|   |                       SYSTEM 1: FAST CONTINUOUS CONTROLLER LOOP (20 - 50 Hz)                        |   |
|   |                                                                                                     |   |
|   |  +-----------------------------+     +-------------------------------+     +---------------------+  |   |
|   |  | Target Memory Tracker       |     | In-Loop Depth-Odometry        |     | Active 2-Axis Gimbal|  |   |
|   |  | - Edge Proximity Servoing   |     | Distance Regulator            |     | Servoing            |  |   |
|   |  | - Direct 3D Re-acquisition  |     | - d_stop = v^2/(2*a) + v*t_r  |     | pitch=atan2(h, d)   |  |   |
|   |  | - Budgeted Search Fallback  |     | - Anti-Overshoot Braking      |     | yaw=atan2(dy, dx)   |  |   |
|   |  +-----------------------------+     +-------------------------------+     +---------------------+  |   |
|   |                 |                                    |                                |             |   |
|   |                 +------------------------------------+--------------------------------+             |   |
|   |                                                      |                                              |   |
|   |                                                      v                                              |   |
|   |                                      +-------------------------------+                              |   |
|   |                                      | 3D Artificial Potential Field |                              |   |
|   |                                      | & Ray-Clamping Collision Core |                              |   |
|   |                                      +-------------------------------+                              |   |
|   |                                                      |                                              |   |
|   |                                                      v                                              |   |
|   |                                      +-------------------------------+                              |   |
|   |                                      | Continuous 6-DoF Kinematics   |                              |   |
|   |                                      | (v_cmd, yaw_rate, altitude)   |                              |   |
|   |                                      +-------------------------------+                              |   |
|   +-----------------------------------------------------------------------------------------------------+   |
|                                                          |                                                  |
|                                                          v                                                  |
|   +-----------------------------------------------------------------------------------------------------+   |
|   |                                   UAV HARDWARE / SIMULATOR INTERFACE                                |   |
|   |   Continuous 6-DoF (AirSim / Webots / PX4 MAVLink) <---> Pluggable OpenFly Benchmark Adapter       |   |
|   +-----------------------------------------------------------------------------------------------------+   |
+=============================================================================================================+
```

---

## 2. Mathematical Formulation & System Foundations

### 2.1 Aerial VLN as a Partially Observable Markov Decision Process (POMDP)
The Aerial VLN problem is formulated as a continuous-space POMDP defined by the 8-tuple:
$$\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R}, \Omega, \mathcal{O}, \mathcal{I}, \gamma \rangle$$

- **State Space $\mathcal{S}$**: The true physical state of the UAV and environment:
  $$\mathbf{s}_t = [\mathbf{p}_t, \mathbf{v}_t, \mathbf{q}_t, \boldsymbol{\omega}_t, \mathcal{E}_t]^T \in \mathcal{S}$$
  where $\mathbf{p}_t = [x, y, z]^T \in \mathbb{R}^3$ is world position, $\mathbf{v}_t = [\dot{x}, \dot{y}, \dot{z}]^T \in \mathbb{R}^3$ is linear velocity, $\mathbf{q}_t \in \mathbb{S}^3$ is attitude quaternion (Euler angles $[\phi, \theta, \psi]$), $\boldsymbol{\omega}_t = [p, q, r]^T \in \mathbb{R}^3$ is body angular rates, and $\mathcal{E}_t$ represents the 3D geometry and semantics of the surrounding environment.
- **Action Space $\mathcal{A}$**: Continuous 6-DoF control inputs:
  $$\mathbf{a}_t = [\mathbf{v}_{\text{cmd}}, \dot{\psi}_{\text{cmd}}, \theta_{g,\text{cmd}}, \psi_{g,\text{cmd}}]^T \in \mathcal{A}$$
  where $\mathbf{v}_{\text{cmd}} \in [-v_{\max}, v_{\max}]^3$, $\dot{\psi}_{\text{cmd}} \in [-\dot{\psi}_{\max}, \dot{\psi}_{\max}]$, and $[\theta_g, \psi_g]$ represent 2-axis gimbal pitch and yaw.
- **Observation Space $\Omega$**: Multimodal observation buffer received at time $t$:
  $$\mathbf{o}_t = \langle \mathbf{I}_t^{\text{RGB}} \in \mathbb{R}^{H \times W \times 3}, \mathbf{D}_t \in \mathbb{R}^{H \times W}, \hat{\mathbf{p}}_t \in \mathbb{R}^3, \hat{\mathbf{v}}_t \in \mathbb{R}^3, \hat{\psi}_t \in [-\pi, \pi) \rangle$$
- **Language Instruction $\mathcal{I}$**: Natural language instruction string $\mathcal{I} = (w_1, w_2, \dots, w_L)$.
- **Reward / Objective $\mathcal{R}$**: Success indicator $\mathbb{I}(\|\mathbf{p}_{\text{final}} - \mathbf{p}_{\text{target}}\| \le \delta_{\text{arrival}} \wedge \text{Collision} = 0)$ subject to trajectory efficiency (SPL - Success weighted by Path Length).

### 2.2 Dual-Rate Bandwidth Dynamics
The fundamental mismatch between high-rate physical control ($\ge 20\text{ Hz}$) and heavy multimodal foundation model inference ($0.1\text{--}0.5\text{ Hz}$) requires strict decoupling of bandwidths:

$$\Delta t_{\text{fast}} = \frac{1}{f_{\text{fast}}} \in [0.02, 0.05]\text{ s} \quad (20\text{--}50\text{ Hz})$$
$$\Delta t_{\text{slow}} = \frac{1}{f_{\text{slow}}} \in [2.0, 10.0]\text{ s} \quad (0.1\text{--}0.5\text{ Hz})$$

Let $k$ index the slow deliberative steps and $t$ index the fast continuous steps. Between two slow reasoning cycles $k$ and $k+1$, System 1 executes $N_k = \frac{\Delta t_{\text{slow}}}{\Delta t_{\text{fast}}} \approx 40\text{--}200$ high-frequency control cycles, maintaining active collision avoidance, stabilization, and tracking.

---

## 3. System 1: Fast Continuous Loop (High-Rate Reflexive Controller)

System 1 operates deterministically at $20\text{--}50\text{ Hz}$. It accepts high-level spatial directives from System 2 and executes low-level motion planning and visual servoing.

### 3.1 Kinematic Equations of Motion & Velocity Profiling
The UAV flight dynamics are modeled as a second-order 3D kinematic system:
$$\begin{aligned}
\dot{\mathbf{p}}(t) &= \mathbf{v}(t) \\
\dot{\mathbf{v}}(t) &= \mathbf{a}_{\text{cmd}}(t) = \frac{\mathbf{v}_{\text{cmd}}(t) - \mathbf{v}(t)}{\tau_v} \\
\dot{\psi}(t) &= \dot{\psi}_{\text{cmd}}(t)
\end{aligned}$$
where $\tau_v$ is the velocity tracking time constant of the internal flight controller.

To ensure smooth trajectories without motor saturation or excessive aerodynamic jerk, velocity commands are filtered through a trapezoidal/S-curve acceleration limiter:
$$\|\mathbf{a}_{\text{cmd}}(t)\| \le a_{\max}, \quad \left\|\frac{d\mathbf{a}_{\text{cmd}}}{dt}\right\| \le j_{\max}$$

### 3.2 In-Loop Depth-Odometry Metric Distance Regulator
**Problem**: Frontier VLMs cannot accurately gauge physical closeness from monocular RGB images. Relying on VLM outputs for proximity detection causes either premature stops ($15\text{m}$ away) or violent high-speed collisions.

**Solution**: System 1 tightly couples the high-rate float32 metric depth map $\mathbf{D}_t$ with real-time odometry $(\mathbf{p}_t, \mathbf{v}_t)$.
1. **Multi-Zone Spatial Depth Probing**: The camera FOV is divided into an active $3 \times 3$ grid $\mathcal{G} = \{g_{i,j}\}_{i,j=1}^3$. For each sector, the robust depth is computed as the 10th percentile depth (closest obstacle):
   $$d_{i,j} = \text{Percentile}_{10}\left(\left\{ \mathbf{D}_t(u, v) \mid (u, v) \in g_{i,j} \right\}\right)$$
2. **Forward Clearance Horizon**:
   $$d_{\text{frontal}} = \min(d_{2,1}, d_{2,2}, d_{2,3})$$
3. **Dynamic Stopping Distance Calculation**:
   Based on current groundspeed $v = \|\mathbf{v}_t\|$, aerodynamic drag, and maximum deceleration $a_{\text{brake}}$, the required stopping distance is:
   $$d_{\text{stop}}(v) = v \cdot t_{\text{react}} + \frac{v^2}{2 a_{\text{brake}}} + d_{\text{margin}}$$
   where $t_{\text{react}} \approx 0.1\text{ s}$ and $d_{\text{margin}} = 1.5\text{m}$.
4. **Adaptive Velocity Scaling (Closeness Regulation)**:
   The forward velocity command $v_f$ is dynamically governed by the clearance ratio:
   $$v_f = \begin{cases}
   0.0 & \text{if } d_{\text{frontal}} \le d_{\text{margin}} \quad (\text{Emergency Braking}) \\
   v_{\max} \cdot \left(\frac{d_{\text{frontal}} - d_{\text{margin}}}{d_{\text{slowdown}} - d_{\text{margin}}}\right)^\alpha & \text{if } d_{\text{margin}} < d_{\text{frontal}} \le d_{\text{slowdown}} \\
   v_{\text{desired}} & \text{if } d_{\text{frontal}} > d_{\text{slowdown}}
   \end{cases}$$
   where $\alpha = 0.7$ provides smooth progressive deceleration.

### 3.3 Dynamic 3D Artificial Potential Field (APF) with Ray Clamping
Obstacle avoidance is continuously governed by an Artificial Potential Field $\mathbf{F}_{\text{total}} = \mathbf{F}_{\text{att}} + \mathbf{F}_{\text{rep}}$:

1. **Attractive Force** toward System 2's current 3D waypoint $\mathbf{w}_k^* = [x_k^*, y_k^*, z_k^*]^T$:
   $$\mathbf{F}_{\text{att}} = k_{\text{att}} \cdot \frac{\mathbf{w}_k^* - \mathbf{p}_t}{\max(1.0, \|\mathbf{w}_k^* - \mathbf{p}_t\|)}$$
2. **Repulsive Force from Depth Sectors**:
   For each spatial depth sector $(i, j)$ with unit line-of-sight vector $\hat{\mathbf{r}}_{i,j}$ and measured depth $d_{i,j}$:
   $$\mathbf{F}_{\text{rep}}^{i,j} = \begin{cases}
   k_{\text{rep}} \left(\frac{1}{d_{i,j}} - \frac{1}{d_0}\right) \frac{1}{d_{i,j}^2} \cdot (-\hat{\mathbf{r}}_{i,j}) & \text{if } d_{i,j} \le d_0 \\
   \mathbf{0} & \text{otherwise}
   \end{cases}$$
3. **Dynamic Altitude Envelope Clamping**:
   To prevent vertical ceiling/floor collisions in urban canyons:
   $$F_{\text{alt}} = \begin{cases}
   +k_z (z_{\min} - z) & \text{if } z < z_{\min} \\
   -k_z (z - z_{\max}) & \text{if } z > z_{\max} \\
   0 & \text{otherwise}
   \end{cases}$$
4. **Ray Clamping**: The resultant command velocity vector $\mathbf{v}_{\text{cmd}} = \mathbf{F}_{\text{total}}$ is projected and clamped against the minimum measured frontal depth ray:
   $$\|\mathbf{v}_{\text{cmd}}\| \le \min\left(v_{\max}, \frac{d_{\text{frontal}} - d_{\text{margin}}}{\Delta t_{\text{fast}}}\right)$$

### 3.4 Active 2-Axis Gimbal Pitch/Yaw Tracking
To keep landmarks and approach corridors centered in the camera's high-resolution center cone without requiring erratic drone yaw turns:
- **Gimbal Pitch Servoing**: Given the relative 3D displacement to the active landmark $(\Delta x, \Delta y, \Delta z)$ in body frame:
  $$d_{\text{horiz}} = \sqrt{\Delta x^2 + \Delta y^2}$$
  $$\theta_{g,\text{desired}} = -\text{atan2}(\Delta z, d_{\text{horiz}})$$
  Clamped to physical joint limits $\theta_g \in [0.0, 0.8\text{ rad}]$ ($0.0 = \text{horizon}$, $0.8 \approx 45.8^\circ\text{ down}$).
- **Gimbal Yaw Servoing**: If the airframe features an actuated gimbal yaw axis:
  $$\psi_{g,\text{desired}} = \text{atan2}(\Delta y, \Delta x)$$
  Otherwise, coordinated airframe yaw tracking aligns the drone's heading:
  $$\dot{\psi}_{\text{cmd}} = k_{\psi} \cdot \text{atan2}(\Delta y, \Delta x)$$

### 3.5 Real-Time Visual Tracking & Zero-Shot Lightweight Model Integration
Between slow VLM cycles ($0.1\text{--}0.5\text{ Hz}$), the fast loop tracks the bounding box of the active navigation target at $30\text{ Hz}$:
- **Tracking Algorithm**: Discriminative Correlation Filter with Channel and Spatial Reliability (CSRT) or lightweight ByteTrack running on RGB stream.
- **Lightweight Zero-Shot Detector Hook**: When available, a local edge model (such as YOLO-World or NanoOWL, quantized to INT8/FP16) processes frames at $15\text{--}30\text{ Hz}$ to detect open-vocabulary text queries emitted by System 2 (e.g., `"red brick chimney"`, `"metal mast"`).
- **Measurement Output**: High-rate normalized target center $[u_t, v_t] \in [-1.0, 1.0]^2$ and bounding box scale $s_t$.

---

## 4. System 2: Slow Reasoning Loop (Deliberative Cognitive Backbone)

System 2 is the intelligent core of the agent. It runs asynchronously, processing high-level semantic tokens and maintaining spatial awareness.

### 4.1 Multimodal Task Decomposition & Dynamic Intent Arbitration
**Critique of Iteration 4 FSM**: Iteration 4 used hardcoded states (`SearchPhase` $\to$ `NavigatePhase` $\to$ `ManeuverPhase`). This breaks down when an instruction requires:
- Conditional exploration (*"If the gate is closed, circle around the perimeter until you find the side entrance"*).
- Backtracking (*"Return to the brick building and face south"*).
- Multi-target sequential traversal.

**Iteration 6 Paradigm**: Elimination of rigid progression state machines. The VLM dynamically generates:
1. **Semantic Intent**:
   - `INTENT_SEARCH_AND_ORIENT`: Searching for ungrounded landmarks described in instruction.
   - `INTENT_CORRIDOR_TRANSIT`: Moving along a flight path or geographic feature (street, river, treeline).
   - `INTENT_TARGET_APPROACH`: Closing distance toward a confirmed 3D scene graph landmark.
   - `INTENT_SURVEY_INSPECT`: Orbiting or performing close visual inspection of a terminal landmark.
   - `INTENT_TERMINAL_VERIFY`: Final visual verification of goal criteria and issuing completion stop.
2. **Subgoal Directives Emitted to System 1**:
   $$\mathcal{U}_k = \langle \text{Intent}_k, \mathbf{w}_k^* \in \mathbb{R}^3, \text{TargetID}_k, v_{\text{target}}, \text{CautionLevel}, \text{TerminationCondition} \rangle$$

### 4.2 Compulsory Feedback Loop & Keyframe Triggers
System 2 is **compulsorily part of the feedback loop; it cannot be bypassed or overridden**.
System 2 is invoked whenever any of the following triggers fire:
1. **Spatial Progress Trigger**: The UAV has traversed a Euclidean distance $\Delta d \ge d_{\text{trigger}}$ (default $8.0\text{m}$) since the last VLM query.
2. **Angular Displacement Trigger**: The UAV has rotated by $|\Delta\psi| \ge 45^\circ$, exposing a novel visual field of view.
3. **Subgoal Arrival Trigger**: System 1 signals that the current 3D waypoint $\mathbf{w}_k^*$ has been reached ($\|\mathbf{p}_t - \mathbf{w}_k^*\| \le \epsilon_{\text{arrival}}$).
4. **Scene Graph Mutation Trigger**: A novel high-confidence landmark node has been discovered, or an active target node has changed visibility state.
5. **Periodic Safety Heartbeat**: Elapsed time $\Delta t \ge 5.0\text{ s}$ without a VLM update.

---

## 5. Zero-Shot 3D Metric Scene Graph & SLAM Anchoring

### 5.1 The Generalizability Imperative: Eliminating Prior Volumetric Dictionaries
In real-world zero-shot VLN on unseen cities, an autonomous agent does *not* know the physical dimensions of buildings, facades, monuments, or vegetation in advance. 

**Prior Defect (Iteration 4)**:
```python
# FAILS ON UNSEEN CITIES:
OBJECT_3D_VOLUMES = {"house": (10.0, 8.0, 6.0), "car": (4.5, 2.0, 1.5)}
pinhole_dist = real_obj_height / (2.0 * math.tan(bbox_scale * vfov / 2.0))
```
If an unseen "manor" is 14m high instead of 7m, the pinhole trigonometric estimate errors by $100\%$, corrupting the entire coordinate map.

**Iteration 6 Solution: Direct Metric Backprojection from Sensor Depth & SLAM Points**:
Let an object detection provide a 2D bounding box $\mathcal{B} = [u_{\min}, v_{\min}, u_{\max}, v_{\max}]$ on the image plane of dimensions $W \times H$.
Let the camera intrinsic matrix be:
$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$
And let the 6-DoF camera pose in world coordinates be $\mathbf{T}_{w,c} = [\mathbf{R}_{w,c} \mid \mathbf{t}_{w,c}] \in \text{SE}(3)$.

1. **Pixel Point Cloud Backprojection**:
   For every valid depth pixel $(u, v) \in \mathcal{B}$ where $D(u, v) \in [d_{\min}, d_{\max}]$:
   $$\mathbf{P}_c(u, v) = D(u, v) \cdot \mathbf{K}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$
   $$\mathbf{P}_w(u, v) = \mathbf{R}_{w,c} \mathbf{P}_c(u, v) + \mathbf{t}_{w,c}$$
2. **Robust Statistical 3D Centroid**:
   To eliminate foreground occlusions and background penetration, the 3D centroid $\mathbf{c} = [X, Y, Z]^T$ is computed as the element-wise median:
   $$\mathbf{c} = \text{Median}\left( \{\mathbf{P}_w(u, v) \mid (u, v) \in \mathcal{B}\} \right)$$
3. **Environment-Agnostic Physical Bounding Extents**:
   The physical width, depth, and height of the object are derived directly from the distribution of $\mathbf{P}_w$ along its principal components:
   $$s_x = \text{Quantile}_{95}(X_w) - \text{Quantile}_{05}(X_w)$$
   $$s_y = \text{Quantile}_{95}(Y_w) - \text{Quantile}_{05}(Y_w)$$
   $$s_z = \text{Quantile}_{95}(Z_w) - \text{Quantile}_{05}(Z_w)$$
   **Result**: Zero dependence on predefined height/width tables. The system measures real physical geometry directly from the sensor data.

### 5.2 SLAM Spatial Anchoring & Map Memory
When dense depth is noisy or degraded (e.g. distant landmarks $>40\text{m}$), the Monocular SLAM Interface provides sparse metric spatial anchoring:
- **Feature Extraction**: Extracts up to 500 ORB keypoints per frame.
- **Epipolar Triangulation**: Triangulates 3D points $\mathbf{X}_i \in \mathbb{R}^3$ across successive keyframes anchored to high-accuracy UAV telemetry (EKF odometry).
- **Global Map Memory**: Maintains a rolling spatial point cloud of up to 8,000 landmark anchor points.
- **Bounding Box Query**: For any bounding box $\mathcal{B}$, queries all active SLAM points projecting inside $\mathcal{B}$, yielding robust ground-truth metric depth even in specular or low-texture conditions.

### 5.3 Spatial Deduplication & Hallucination Filtering
To prevent multiple detections of the same building from cluttering the graph as distinct entities:
1. **Mahalanobis / Euclidean Distance Gating**:
   A newly observed object with centroid $\mathbf{c}_{\text{new}}$ is matched against existing scene graph nodes $\{\mathcal{N}_j\}$:
   $$d_j = \|\mathbf{c}_{\text{new}} - \mathbf{c}_j\|$$
   The match radius is dynamically scaled by the estimated object size:
   $$R_{\text{match}} = \max\left(2.5\text{m}, 0.5 \cdot \max(s_{x,j}, s_{y,j})\right)$$
2. **Semantic Similarity Gating**:
   If $d_j \le R_{\text{match}}$ and cosine similarity $\cos(\mathbf{e}_{\text{label}}, \mathbf{e}_j) \ge 0.65$:
   Associate the detection with node $j$.
3. **Bayesian Centroid Fusion**:
   $$\mathbf{c}_j^{(k+1)} = \mathbf{c}_j^{(k)} + \mathbf{K}_k \left(\mathbf{c}_{\text{new}} - \mathbf{c}_j^{(k)}\right)$$
   where Kalman gain $\mathbf{K}_k = \frac{\mathbf{P}_j}{\mathbf{P}_j + \mathbf{R}_{\text{sensor}}}$.
4. **Anti-Hallucination Confidence Filter**:
   Every node maintains an observation count $N_{\text{obs}}$ and confidence score:
   $$\mathcal{C}_j = 1 - e^{-\lambda N_{\text{obs}}}$$
   - *Transient Candidate*: $N_{\text{obs}} = 1$. Displayed with low confidence; not used for primary waypoint generation.
   - *Confirmed Grounded Landmark*: $N_{\text{obs}} \ge 2$ observed from distinct viewpoints ($\|\mathbf{p}_{\text{uav}, 1} - \mathbf{p}_{\text{uav}, 2}\| > 1.5\text{m}$). Guaranteed real landmark.

### 5.4 Structured Spatial Prompt Serialization for VLM
At each System 2 step, the confirmed 3D Scene Graph is serialized into structured, token-efficient text for the VLM's context prompt:

```markdown
[CONFIRMED 3D SCENE GRAPH - METRIC WORLD FRAME]
UAV Current State: Pose=(X=12.4m, Y=28.7m, Z=10.0m) | Heading=042° (NE) | Speed=2.1m/s
Active Landmarks (Deduplicated & Grounded):
- ID #01 [manor house]: World(X=24.1, Y=45.2, Z=6.5) | Relative: Dist=20.5m, Bearing=+11°, Elev=-10° | Dim: (12.2m x 9.8m x 7.4m) | Conf=0.96 [CONFIRMED]
- ID #02 [wind turbine mast]: World(X=42.8, Y=31.0, Z=14.0) | Relative: Dist=30.5m, Bearing=-38°, Elev=+07° | Dim: (3.1m x 3.1m x 15.0m) | Conf=0.92 [CONFIRMED]
- ID #03 [flat roof solar array]: World(X=35.0, Y=62.0, Z=5.2) | Relative: Dist=40.2m, Bearing=+24°, Elev=-07° | Dim: (8.0m x 6.5m x 1.2m) | Conf=0.74 [CONFIRMED]
```
This grounds the VLM strictly to physical Euclidean space, eliminating speculative hallucinations.

---

## 6. Target Memory Tracker & Active Recovery (Replacing Coasting Mode)

### 6.1 Edge Proximity Detection & Anticipatory Servoing
The Target Memory Tracker monitors the normalized image position of the active target $(u_t, v_t) \in [-1.0, 1.0]^2$:
- **Safe Zone**: $|u_t| \le 0.65$ and $|v_t| \le 0.65$. The target is securely centered in the visual cone.
- **Edge Boundary Trigger**:
  $$\text{Edge Proximity} = \begin{cases}
  \text{LEFT\_EDGE} & \text{if } u_t < -0.70 \\
  \text{RIGHT\_EDGE} & \text{if } u_t > +0.70 \\
  \text{TOP\_EDGE} & \text{if } v_t < -0.70 \\
  \text{BOTTOM\_EDGE} & \text{if } v_t > +0.70
  \end{cases}$$
- **Lead Servoing**: When an edge trigger fires, System 1 applies an anticipatory centering velocity *before* the target leaves the frame:
  $$\dot{\psi}_{\text{lead}} = -k_\psi \cdot u_t, \quad \dot{\theta}_{g,\text{lead}} = -k_\theta \cdot v_t$$

### 6.2 Elimination of Coasting Mode: Metric 3D Scene Graph Guided Re-Acquisition
In Iteration 4, when visual detection was lost, the drone entered a 3-second coasting mode, locking body translations and blindly sweeping gimbal pitch.

**Iteration 6 Architectural Replacement**:
When an active target exits visual detection:
1. **Target Position Retrieval**:
   System 1 immediately queries the 3D Scene Graph for the target's last confirmed global coordinates $\mathbf{c}_{\text{target}} = [X_t, Y_t, Z_t]^T$.
2. **Closed-Loop Vector Navigation**:
   Instead of coasting or freezing, the UAV computes the 3D re-acquisition vector:
   $$\hat{\mathbf{v}}_{\text{reacq}} = \frac{\mathbf{c}_{\text{target}} - \mathbf{p}_{\text{uav}}}{\|\mathbf{c}_{\text{target}} - \mathbf{p}_{\text{uav}}\|}$$
   System 1 flies directly toward $\mathbf{c}_{\text{target}}$ while continuously pointing the camera toward the target's estimated location using inverse kinematics:
   $$\psi_{\text{cam}} = \text{atan2}(Y_t - y_{\text{uav}}, X_t - x_{\text{uav}})$$
   $$\theta_{g,\text{cam}} = -\text{atan2}(Z_t - z_{\text{uav}}, \sqrt{(X_t - x_{\text{uav}})^2 + (Y_t - y_{\text{uav}})^2})$$
3. **Budgeted Local Search Sweep**:
   If the UAV arrives at $\mathbf{c}_{\text{target}}$ ($\|\mathbf{p}_{\text{uav}} - \mathbf{c}_{\text{target}}\| \le 4.0\text{m}$) and the target has not been visually re-acquired (due to dynamic movement or occlusion):
   - The drone initiates a **Budgeted Local Scan**: A maximum of $N_{\text{scan}} = 6$ angular yaw steps ($\Delta\psi = 60^\circ$) while varying gimbal pitch between $0.2\text{ rad}$ and $0.5\text{ rad}$.
   - If re-acquired: Resume normal continuous approach.
   - If budget exhausted ($N_{\text{scan}} \ge 6$ without detection): Declare `TARGET_LOST_CONFIRMED`. System 1 hovers safely and immediately triggers an asynchronous System 2 deliberative re-planning cycle.

---

## 7. Safety-Only Fallback State Machine

In Iteration 6, **Finite State Machine (FSM) logic is strictly demoted from mission progression to low-level emergency and safety handling**. The high-level flight progression is dictated entirely by System 2's VLM reasoning.

### Fallback State Architecture
```
                         +-----------------------------+
                         |     STATE 0: NORMAL         |
                         |  (System 1 executes S2 sub- |
                         |   goals with APF avoidance) |
                         +-----------------------------+
                            |          ^          |
       Obstacle clearance   |          | Clear    | Target visual track
       d_frontal < d_margin |          | horizon  | lost from FOV
                            v          |          v
          +-----------------------+  +-------------------------------+
          |  STATE 1: EMERGENCY   |  |   STATE 2: 3D RE-ACQUISITION  |
          |  OBSTACLE REPULSION   |  |  (Fly to Scene Graph coords; |
          |  (Hard brake + ascend)|  |   point gimbal at target)     |
          +-----------------------+  +-------------------------------+
                                                  | Target missing at
                                                  | target coordinates
                                                  v
                                     +-------------------------------+
                                     |   STATE 3: BUDGETED SCAN      |
                                     |  (Max 6 sector turns; search  |
                                     |   budget enforcement)         |
                                     +-------------------------------+
                                                  | Search budget exhausted
                                                  v
                                     +-------------------------------+
                                     |   STATE 4: SAFE HOVER & S2    |
                                     |   RE-PLANNING TRIGGER         |
                                     +-------------------------------+
```

- **State 0: NORMAL_OPERATION**: Fast continuous controller drives the UAV along System 2's subgoals while APF and in-loop depth regulator maintain dynamic clearance.
- **State 1: EMERGENCY_OBSTACLE_REPULSION**: Fired when $d_{\text{frontal}} \le 1.5\text{m}$. Overrides forward commands, applies maximum reverse deceleration, and initiates vertical climb ($\Delta z = +1.5\text{m}$) until clearance is restored.
- **State 2: TARGET_REACQUISITION**: Executes metric flight to the last known 3D Scene Graph centroid of the lost target.
- **State 3: BUDGETED_SEARCH**: Executes bounded active angular search (budget $\le 6$ steps).
- **State 4: HOVER_AND_REPLAN**: Failsafe hover state invoked when target cannot be found or communication with System 2 times out ($\Delta t > 10\text{s}$).

---

## 8. Decoupled Kinematics & Pluggable Benchmark Adapters

The core framework is formulated for **general-purpose, continuous 6-DoF aerial navigation** with an actuated camera gimbal. To support benchmark evaluations (e.g. OpenFly) without corrupting the general architecture, all hardware/benchmark constraints are encapsulated into **Pluggable Adapters**.

### 8.1 Core Flight Controller Interface
```python
class ContinuousUAVController(ABC):
    """General-purpose continuous 6-DoF UAV interface."""
    @abstractmethod
    def set_continuous_velocity(self, vx: float, vy: float, vz: float, yaw_rate: float) -> None:
        """Command continuous velocity in body or world frame."""
        pass

    @abstractmethod
    def set_gimbal_angles(self, pitch: float, yaw: float) -> None:
        """Command continuous 2-axis gimbal attitude."""
        pass

    @abstractmethod
    def get_telemetry(self) -> DroneTelemetry:
        """Return 6-DoF pose (x, y, z, roll, pitch, yaw) and linear/angular velocity."""
        pass
```

### 8.2 OpenFly Benchmark Adapter Layer
The OpenFly benchmark imposes two artificial hardware constraints:
1. Fixed forward-facing camera ($\theta_g = 0^\circ$).
2. 10 discrete integer action codes (`MOVE_FORWARD_3M`, `TURN_LEFT_30`, etc.).

The **OpenFly Adapter** translates continuous controller outputs into the benchmark's discrete space:
```python
class OpenFlyBenchmarkAdapter:
    """Translates continuous System 1 directives into OpenFly discrete action tokens."""
    def quantize_action(self, v_cmd: np.ndarray, yaw_rate_cmd: float, dt: float) -> int:
        dx = v_cmd[0] * dt
        dy = v_cmd[1] * dt
        dz = v_cmd[2] * dt
        dyaw_deg = math.degrees(yaw_rate_cmd * dt)

        # Priority 1: Large heading adjustments
        if dyaw_deg > 15.0:
            return ACTION_TURN_LEFT_30
        elif dyaw_deg < -15.0:
            return ACTION_TURN_RIGHT_30
        
        # Priority 2: Vertical clearance
        if dz > 1.5:
            return ACTION_CLIMB_2M
        elif dz < -1.5:
            return ACTION_DESCEND_2M
        
        # Priority 3: Forward progression
        if math.hypot(dx, dy) >= 1.5:
            return ACTION_FORWARD_3M
        
        return ACTION_HOVER
```
This cleanly isolates benchmark idiosyncrasies, allowing the same agent core to fly in Webots, AirSim, Isaac Sim, or real PX4 quadrotors.

---

## 9. Low-Level Implementation Plan & Module Hierarchy

### 9.1 Module Directory Structure
```
controllers/mavic_vlm_controller/
├── core/
│   ├── __init__.py
│   ├── kinematics.py         # 6-DoF continuous kinematic integrator & APF engine
│   ├── depth_regulator.py    # Odometry-in-the-loop multi-zone depth monitor
│   ├── scene_graph_3d.py     # Environment-agnostic 3D Scene Graph & deduplication
│   ├── slam_interface.py     # 500-point ORB feature tracker & 8000-point world memory
│   └── tracker_memory.py     # Target memory tracker, edge proximity & active recovery
├── cognitive/
│   ├── __init__.py
│   ├── vlm_client.py         # Multi-backend VLM client (Qwen-VL, GPT-4o, Claude 3.5)
│   ├── intent_arbiter.py     # Open-ended instruction decomposer & subgoal generator
│   └── prompt_grounder.py    # 3D Scene Graph spatial serializer
├── adapters/
│   ├── __init__.py
│   ├── base_adapter.py       # Abstract continuous UAV interface
│   ├── openfly_adapter.py    # OpenFly discrete & fixed-camera test harness
│   ├── airsim_adapter.py     # AirSim RPC continuous visual servoing interface
│   └── webots_adapter.py     # Webots Mavic 2 Pro continuous supervisor
└── main_controller.py        # Master dual-rate asynchronous execution runtime
```

### 9.2 Key Data Structures & Method Signatures

#### `scene_graph_3d.py`
```python
@dataclass
class SceneNode:
    node_id: int
    label: str
    centroid_world: np.ndarray  # [X, Y, Z] in meters
    dimensions_3d: np.ndarray   # [length, width, height] in meters
    confidence: float
    observation_count: int
    last_observed_timestamp: float
    first_viewpoint: np.ndarray
    is_confirmed: bool

class MetricSceneGraph3D:
    def __init__(self, match_radius_base: float = 2.5, min_views_to_confirm: int = 2):
        self.nodes: Dict[int, SceneNode] = {}
        self.next_node_id = 1
        self.match_radius_base = match_radius_base
        self.min_views_to_confirm = min_views_to_confirm

    def add_or_update_detection(
        self,
        label: str,
        bbox: Tuple[float, float, float, float],  # [u_min, v_min, u_max, v_max]
        depth_map: np.ndarray,
        camera_intrinsics: np.ndarray,            # 3x3 K matrix
        camera_pose_world: np.ndarray,            # 4x4 SE(3) matrix
        uav_position: np.ndarray,
        timestamp: float,
        slam_points: Optional[List[np.ndarray]] = None
    ) -> SceneNode:
        """Backprojects BBox to 3D, computes median centroid and extent, and updates graph."""
        ...
```

#### `tracker_memory.py`
```python
class TargetRecoveryState(Enum):
    TRACKING = "TRACKING"
    EDGE_ANTICIPATION = "EDGE_ANTICIPATION"
    METRIC_3D_REACQUISITION = "METRIC_3D_REACQUISITION"
    BUDGETED_SEARCH = "BUDGETED_SEARCH"
    LOST_DECLARED = "LOST_DECLARED"

class TargetMemoryTrackerV6:
    def __init__(self, search_step_budget: int = 6):
        self.state = TargetRecoveryState.TRACKING
        self.active_target_id: Optional[int] = None
        self.target_centroid_3d: Optional[np.ndarray] = None
        self.search_steps_taken = 0
        self.search_step_budget = search_step_budget

    def update_visual_tracking(
        self,
        target_visible: bool,
        bbox_norm: Optional[Tuple[float, float, float, float]],
        scene_graph: MetricSceneGraph3D
    ) -> TargetRecoveryAction:
        """Determines whether to servo, execute 3D trajectory to last known coordinates, or scan."""
        ...
```

---

## 10. Scientific Contribution & Architectural Comparison Matrix

| Architectural Feature | Iteration 1 (Baseline) | Iteration 3 (Spatio-Temporal) | Iteration 4 (OmniNav Fast-Slow) | Iteration 5 (CalibratedNav) | **Iteration 6 (Proposed)** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cognitive Paradigm** | Discrete Policy | Monolithic Transformer | Decoupled Fast-Slow | Closed-Loop Visual Tooling | **Decoupled Dual-System (Fast Continuous + Slow Deliberative)** |
| **Control Action Space** | 10 Discrete Actions | Discrete Action Codes | Continuous APF with Discrete Quantizer | Step Continuous ($\Delta x, \Delta y, \Delta z$) | **Continuous 6-DoF ($\mathbf{v}_{\text{cmd}}, \dot{\psi}, \theta_g, \psi_g$) with Pluggable Adapters** |
| **Depth & Closeness Perception** | None (Monocular RGB) | Periodic Sensor Fetch | Monocular Pinhole / SLAM | 9-Point Spatial Grid Probing | **Continuous In-Loop Odometry-Depth Dynamic Braking Regulator** |
| **3D Object Position Mapping** | None | 2D BEV Grid ($64\text{m}\times 64\text{m}$) | Hardcoded Prior Heights (`house: 6m`) | Pinhole Deprojection ($u^*, v^*, D^*$) | **Zero-Shot 3D Metric Scene Graph (Median Centroid + PCA Extents)** |
| **Target Loss Handling** | Open-Loop Drift | None | 3-Second Blind Coasting & Lock | Step-by-Step Re-Query | **Closed-Loop 3D Re-Acquisition to Graph Centroid + Budgeted Scan** |
| **Mission State Management** | None | Implicit Transformer Hidden State | Rigid FSM (`Search` $\to$ `Navigate` $\to$ `Maneuver`) | Instruction Chunk Boundaries | **Open-Ended VLM Intent Arbitration; FSM Restricted to Safety Fallbacks** |
| **Gimbal Integration** | Fixed 0° | Fixed 0° | Dynamic Pitch Profile ($0.0 - 0.8\text{ rad}$) | Approximate Vertical Elevation $v^*$ | **Active 2-Axis Continuous Gimbal Tracking ($\text{pitch}=\text{atan2}(h, d)$)** |
| **Generalizability to Unseen Cities** | Poor | Low (Fixed Map Bounds) | Poor (Hardcoded Volumes & Profiles) | Moderate | **High (Zero hardcoded volume or search dictionaries)** |

---

## 11. Open-Source Component Integration (Attribution & Code Reuse)

In alignment with user instructions, the framework integrates proven, high-performance open-source modules without adopting proprietary research ideation:
1. **Visual Keypoint Tracking & SLAM Backend**:
   - Integrated component from `pySLAM` / `ORB-SLAM3` Python bindings (`cv2.ORB_create`, `cv2.calcOpticalFlowPyrLK`).
   - Reused for robust 3D keypoint triangulation and world-frame spatial anchoring.
2. **Real-Time Visual Object Tracker**:
   - OpenCV `TrackerCSRT` / `ByteTrack` (`from yolox.tracker.byte_tracker import BYTETracker`).
   - Reused for high-rate ($30\text{ Hz}$) bounding-box tracking between slow VLM reasoning cycles.
3. **Spatial Visualization & Telemetry**:
   - `Rerun.io` SDK (`import rerun as rr`).
   - Native 3D logging of point clouds, camera frustums, scene graph nodes, and UAV trajectory.

---

## 12. Verification & Experimental Protocol

The Iteration 6 framework will be evaluated across two rigorous benchmark tiers to validate both general aerial navigation and benchmark compliance:

1. **Continuous 6-DoF General Aerial VLN Benchmark**:
   - Environments: High-fidelity photo-realistic urban scenes (AirSim / Webots city worlds) with unmapped skyscrapers, narrow alleys, varying building elevations, and dynamic lighting.
   - Metrics: Success Rate ($\text{SR}_{5.0\text{m}}$, $\text{SR}_{20.0\text{m}}$), Oracle Success Rate (OSR), Navigation Error (NE in meters), Success weighted by Path Length (SPL), Collision Rate (CR), and Camera Centering Tracking Index (CCTI).
2. **OpenFly Benchmark Adapter Suite**:
   - Evaluation across the 40 standard OpenFly test episodes using `OpenFlyBenchmarkAdapter`.
   - Comparison directly against Iteration 1, Iteration 4, and official AgenticNav/OpenFly baselines.
