# Failure Mode Taxonomy & Diagnosis
## Aerial Vision-and-Language Navigation

To ensure complete transparency and prevent regression, every benchmark failure is categorized and diagnosed under this taxonomy.

---

### Category A: Heading & Yaw Drift
- **Symptom**: High Navigation Error ($\text{NE} > 80\text{m}$) from step 1; drone flies radially away from destination.
- **Root Causes**:
  1. *Uninitialized Start Heading*: Setting $\theta_0 = 0.0$ instead of extracting $\text{yaw}_0$ from the dataset (`raw_yaws[0]`). (Resolved in Refined Framework).
  2. *Turn Discretization Accumulation*: Iterated discrete $30^\circ$ turns over- or under-shoot complex angular bearings (e.g., $45^\circ$ or $75^\circ$), compounding into cross-track drift.
- **Remediation**: Closed-loop visual visual servoing aligning drone centerline with visual landmark bounding boxes.

---

### Category B: Premature Termination & Overshoot (Arrival Gating)
- **Symptom**: Large gap between Oracle Success Rate (OSR) and Success Rate (SR) (e.g. $\text{OSR}=22.5\%$ vs $\text{SR}=12.5\%$).
- **Root Causes**:
  1. *Fly-By Without Stopping*: The drone reaches a closest approach of $12\text{--}18\text{m}$, but does not trigger `STOP` because the visual landmark is still visible in the distance.
  2. *Plan Exhaustion Before Arrival*: The action queue empties before reaching the threshold, triggering an emergency stop in open space.
- **Remediation**: Dual-gated arrival:
  - Trigger `STOP` when forward metric depth $d \le 3.5\text{m}$.
  - Or trigger `STOP` when Euclidean distance along Kalman-filtered landmark track falls within arrival cylinder.

---

### Category C: Altitude Drift & Roof Clamping
- **Symptom**: Drone flies over goal building at high altitude without descending, or collides with building parapet.
- **Root Causes**:
  1. Dataset records positive $z$ as climb, while AirSim Unreal NED coordinates use negative $z$ for altitude above ground. Sign inversions cause inverted altitude maneuvers.
  2. Altitude envelope not bounded: drone climbs on every obstacle encounter but fails to descend into terminal approach zone.
- **Remediation**: Bounded altitude corridors ($z_{\text{min}} = -1.5\text{m}$, $z_{\text{max}} = -25\text{m}$ in NED) with relative elevation tracking.

---

### Category D: Model Inference Timeouts & Format Hallucinations
- **Symptom**: Episode halts or takes $> 45\text{s}$ per step; VLM outputs conversational text without valid action codes.
- **Root Causes**:
  1. Running heavy 8B/9B VLMs on local low-VRAM laptops causes KV-cache swapping and timeout.
  2. Unconstrained generation allows the model to output discursive thinking tokens without valid JSON.
- **Remediation**: Strict execution on remote NVIDIA L40S server with regex JSON extraction and persistent VRAM caching (`keep_alive: -1`).

---

### Category E: Clause Fragmentation in Instruction Chunking
- **Symptom**: Model enters unbounded reasoning loop (> 15,000 thinking characters), exhausts generation token budget, and returns empty text.
- **Root Cause**: Naive splitting on `\band\b` destroys compound adjectives (e.g. `"green and brown"` becomes `"green"` and `"brown medium - sized area..."`). Verbless fragments lack motion intent, causing reasoning models to loop indefinitely.
- **Remediation**: Semantic regex splitting that only cuts on `and` when followed by motion verbs:
  `r'[;.\n]|\b(?:then|next|finally|after that)\b|\band\s+(?=(?:fly|turn|head|proceed|advance|move|climb|descend|go)\b)'`.

---

### Category F: Institutional Cluster Session Quota Exhaustion
- **Symptom**: Sudden HTTP 503 Service Unavailable or HTTP 409 Conflict: `"Your 5 h session time is used up. A new session is available at 15:55."`
- **Root Cause**: Remote institutional GPU server imposes hard 5-hour session ceilings per container to ensure fair resource allocation.
- **Remediation**:
  1. Log live telemetry to detached files (`benchmark_40ep_live.log`) on persistent `/home/jovyan/scratch/` volume.
  2. Implement automatic session renewal hooks (`POST /services/gpu-dashboard/api/session/start`) upon quota window reset.
  3. Continuous local backup synchronization via `sync_from_remote.py` to protect all trajectory artifacts.

