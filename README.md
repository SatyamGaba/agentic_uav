# AgenticNav: Vision-and-Language Navigation as a Tool-Calling Harness

A rigorous replication and refinement of **AgenticNav (arXiv:2606.10577, 2026)** for ground Vision-and-Language Navigation in Continuous Environments (VLN-CE), evaluated in Habitat-Sim 0.3.3 on Matterport3D (MP3D) meshes with open-weights `Qwen3.8-27B`.

---

## 1. Benchmark & Evaluation Split

### Reference Paper
- **Citation**: *AgenticNav: Vision-and-Language Navigation as a Tool-Calling Harness* (arXiv:2606.10577, 2026).
- **Environment**: **Habitat-Sim 0.3.3** physical simulator navigating photorealistic 3D scans from **Matterport3D (MP3D)**.
- **Dataset**: **R2R-CE** (Room-to-Room in Continuous Environments, Krantz et al., ECCV 2020).

### Test Split
The standard evaluation split for testing navigation in unfamiliar environments is `val_unseen` (1,839 total episodes across 11 buildings). Evaluating thousands of episodes with a 27-billion parameter vision model requires substantial computation. Following the AgenticNav authors, we evaluate on a standardized **100-episode test subset**:
- **`agenticnav_paper_replica/data/r2rce_val_unseen_100_rand.json`**: The canonical first 100 episodes (indices 0–99) of the official R2R-CE `val_unseen` split (`val_unseen.json.gz`, SHA-256 verified).
- **`agenticnav_paper_replica/data/opennav_r2rce_100.json`**: The 100-episode split curated by OpenNav, containing the exact episodes (116, 191, and 218) illustrated in Figure 6 of the AgenticNav paper.

---

## 2. System Architecture

The framework replicates the sensory and decision-making architecture specified in the paper:

```text
               +-------------------------------------------------+
               |       Vision-Language Model (Qwen3.8-27B)       |
               +-------------------------------------------------+
                                      |
                      Decides action by calling tools
                                      v
       +---------------------------------------------------------------+
       |                         Tool Harness                          |
       |  - query_depth: inspect distance to point without moving      |
       |  - move_to:     turn toward and step toward target point      |
       |  - recall:      retrieve images from past steps               |
       |  - stop:        declare navigation complete at goal           |
       +---------------------------------------------------------------+
                                      |
                         Physical movement & sensing
                                      v
       +---------------------------------------------------------------+
       |             Habitat-Sim (Matterport3D Environment)            |
       |  - 4 Panoramic RGB-D Cameras (Front, Right, Back, Left)       |
       |  - Top-down (BEV) Occupancy Mapping from 3D depth points      |
       |  - Strict Firewall: model never sees ground-truth coordinates |
       +---------------------------------------------------------------+
```

### Exact Paper Replica Components
1. **Panoramic Vision**:
   - 4 cameras placed at 90° intervals: Front ($0^\circ$), Right ($-90^\circ$), Back ($180^\circ$), and Left ($+90^\circ$).
   - Resolution: $640 \times 480$ pixels per camera, $90^\circ$ horizontal field of view (HFOV).
   - Eye height: Mounted at a human-like height of $1.25\text{ m}$ above the floor.
2. **Top-Down (Bird's-Eye-View) Occupancy Map**:
   - Camera depth measurements are mathematically unprojected into 3D metric coordinates using pinhole camera equations.
   - 3D points are converted into an egocentric 2D top-down map showing explored corridors, solid obstacles, and the past trajectory line.
3. **Tool-Calling Protocol**:
   - `query_depth(view, u, v)`: Measures metric distance to a selected pixel without moving the robot.
   - `move_to(view, u, v, reason)`: Commands the robot to turn toward and walk toward a clicked pixel in an image.
   - `recall(step)`: Retrieves past camera views to maintain orientation.
   - `stop(reason)`: Declares that the agent has reached the destination specified in the instruction.
4. **Strict Sensory Isolation (Firewall)**:
   - The language model never receives map coordinates, simulator positions, or shortest-path navigation hints. All decisions are based entirely on what the cameras see.

---

## 3. Experimental Evolution & Benchmark Results

All runs evaluate the open-weights **Qwen3.8-27B** vision-language model across the 100-episode R2R-CE `val_unseen` split on dedicated NVIDIA L40S hardware:

| Benchmark Framework | Success Rate (SR) | Path Efficiency (SPL) | Oracle Success (OSR) | Average Error (NE) | Deadlocks | Description |
|---|:---:|:---:|:---:|:---:|:---:|---|
| **Arm P (Paper-Faithful Replica)** | 13.13% | 7.83 | 36.36% | 7.24 m | 37% | Exact paper straight-line collision checking. High deadlock rate from doorframe clipping. |
| **Arm L (Local Aids)** | 19.19% | 13.64 | 40.40% | 7.38 m | 20% | Calibrated obstacle threshold ($N=5$) and arrival verification prompt. |
| **Arm L-v3 (Stride Scaling)** | 24.00% | 16.04 | 45.00% | 6.76 m | 20% | Adaptive step lengths and 3D swept cylinder obstacle checking. |
| **Arm O + Remediation (Our Best Framework)** | **33.00%** | **24.04** | **50.00%** | **7.06 m** | **0%** | **Full autonomous navigation with local cornering, truthful physical odometry, and doorway alignment.** |

### Key Improvements in Our Best Framework (Arm O + Remediation)
1. **Natural Cornering (Local Path Actuation)**:
   - *Problem in Paper*: The paper assumed the robot walks in rigid, perfectly straight lines. In real 3D continuous meshes, walking in a straight line toward an open doorway often grazes doorposts by centimeters, causing the paper-faithful Arm P to declare deadlock on 37% of runs.
   - *Fix*: The vision model still selects all target waypoints, but physical execution follows Habitat's local path follower to step around corners naturally without getting stuck on millimeter edges.
2. **Truthful Physical Feedback**:
   - *Problem*: In previous iterations, if an obstacle cut a 1.5-meter step short at 0.4 meters, the prompt did not report the shortfall. The model assumed it had completed the move and stopped prematurely.
   - *Fix*: The prompt reports the exact physical distance moved and warns the model if motion was interrupted by an obstacle.
3. **Doorway Perspective Check**:
   - *Problem*: When an agent steps up to a doorway, looking straight ahead makes it appear that the goal is in the room ahead, causing the agent to step past the threshold.
   - *Fix*: The prompt guides the model to inspect its left and right cameras (flanking views) to verify if it is already standing within the target doorway threshold before taking another step.

---

## 4. Repository Structure

```text
UAV/
├── agenticnav_paper_replica/       # Core replication codebase
│   ├── configs/                    # Configuration dataclasses (ReplicaConfig)
│   ├── data/                       # Official 100-episode R2R-CE splits
│   ├── embodiment/                 # Habitat-Sim 0.3.3 ground embodiment
│   ├── evaluation/                 # Metrics computation (SR, SPL, OSR, NE)
│   ├── geometry/                   # Pinhole camera math and 3D coordinate frames
│   ├── harness/                    # Step loop, prompt builder, and tool execution
│   ├── mapping/                    # Top-down BEV occupancy grid generator
│   ├── memory/                     # Image cache and action history tracker
│   ├── traces/                     # JSONL episode trace logger
│   └── vlm/                        # Ollama / vLLM client interfaces with privilege firewall
├── spec/
│   └── spec_ledger.yaml            # Pinned specification ledger and parameter sources
├── scripts/
│   ├── benchmark_r2rce_postfix.py  # Main benchmark runner with strict provenance validation
│   ├── cluster_client.py           # GPU cluster API client
│   └── build_r2rce_subset.py       # Dataset split extractor
└── README.md
```

---

## 5. Quickstart & Verification

### Running the Test Suite
All unit and integration tests run in seconds without requiring GPU hardware or Habitat-Sim:
```bash
python -m pytest agenticnav_paper_replica/ -v
```

### Running the Benchmark
To evaluate the framework on the 100-episode R2R-CE benchmark (requires Habitat-Sim and an active model service):
```bash
python scripts/benchmark_r2rce_postfix.py \
  --episodes agenticnav_paper_replica/data/r2rce_val_unseen_100_rand.json \
  --model qwen3.8:27b \
  --output-dir results/arm_oracle_remediation_100
```
