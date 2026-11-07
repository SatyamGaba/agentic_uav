# Remote GPU Cluster Guide (Ananta Hub)

This guide explains how to connect to, monitor, synchronize code with, and execute multi-hour navigation benchmarks on the remote GPU server (`172.24.16.175`).

---

## 1. Network Connectivity & Pre-Flight Check

The cluster is hosted at internal IP `172.24.16.175`. If requests return **`HTTP 403 Forbidden`** with a block page:
1. **On Campus (Wi-Fi / Ethernet)**: Open your browser and log into the BITS Goa campus captive portal:
   ```
   https://campnet.bits-goa.ac.in:8090/
   ```
   *Reason:* Sophos firewall restricts direct IP access until LDAP credentials are authenticated.
2. **Off Campus**: Connect to the campus network via **FortiClient VPN** before running the scripts.

To test network connectivity at any time:
```powershell
python remote/status.py
```

---

## 2. Hardware Architecture & GPU Roles

| Resource | Specification | Role |
| :--- | :--- | :--- |
| **GPU 0 (Session GPU)** | **NVIDIA L40S PCIe** (48 GB GDDR6) | Interactive Jupyter notebook/terminal containers. Subject to daily quota and idle shutdown. |
| **GPU 1 (Job GPU)** | **NVIDIA L40S PCIe** (48 GB GDDR6) | **Dedicated batch job queue**. Completely independent of GPU 0; runs multi-hour jobs continuously without interactive timeouts. |
| **Compute CPU** | Dual Intel Xeon Gold 6526Y (Emerald Rapids) | 32 cores / 64 threads |
| **System RAM** | 132.5 GB DDR5 | ~110 GB free |
| **Scratch Space** | 4.76 TB HDD mounted at `/home/jovyan/scratch/` | Persistent storage across jobs |

---

## 3. End-to-End Workflow for Multi-Hour Benchmark Jobs

### Step 1: Check Environment & Hardware Health
Run the automated remote environment diagnostic:
```powershell
python remote/check_env.py
```
This runs a 5-minute diagnostic on GPU 1 that tests:
* NVIDIA GPU driver, compute capability, and PyTorch CUDA tensor acceleration.
* Installed Python packages (`torch`, `numpy`, `scipy`, `PIL`, `habitat`).
* Free disk space on `/home/jovyan/scratch`.
* Existing datasets and model caches.

---

### Step 2: Transfer All Files to the Remote Server
Package and transfer the entire `agenticnav_paper_replica` repository and runner scripts:
```powershell
python remote/sync_all.py
```
*Destination on remote:* `/home/jovyan/scratch/agenticnav_paper_replica/`
*Automated Verification:* Unpacks the tarball, creates required output directories (`runs/`, `traces/`), imports the package, and runs unit tests directly on the server to guarantee integrity.

*(Optional dry-run preview: `python remote/sync_all.py --dry-run`)*

---

### Step 3: Launch Multi-Hour Continuous Benchmark Job
Launch a continuous execution loop on dedicated **GPU 1**:
```powershell
# Standard 6-hour replication job (K=3 repetitions, all 100 episodes):
python remote/launch_benchmark_job.py --name "agenticnav-replica-full" --time 360 --repetitions 3

# Fast 10-episode smoke-test (1 hour limit):
python remote/launch_benchmark_job.py --name "smoke-test-10ep" --time 60 --max-episodes 10 --repetitions 1

# Ablation experiment without depth tool:
python remote/launch_benchmark_job.py --name "ablation-wo-depth" --ablation wo_depth --time 240
```

**Key Multi-Hour Protections Built In:**
1. **Checkpointing & Auto-Resumption**: Every completed episode is atomically saved to `checkpoint.json`. If the job is re-run or extended, it skips already-completed runs and resumes instantly.
2. **GPU Idle Watchdog Heartbeat**: A lightweight background daemon maintains low-level tensor pulses to keep the cluster watchdog disarmed while the system waits for VLM inferences.
3. **Exception Resilience**: Individual episode failures or API hiccups are safely caught and categorized as `infrastructure_failure` so the multi-hour loop continues uninterrupted.

---

### Step 4: Real-Time Live Progress Tracking
Follow the progress of your running job in real time:
```powershell
# Auto-detects the active benchmark job and opens the live dashboard:
python remote/track_progress.py

# Or monitor a specific Job ID:
python remote/track_progress.py --job-id <JOB_ID>
```

**Live Dashboard Output:**
```
========================================================================
         ANANTA REMOTE GPU CLUSTER: LIVE BENCHMARK MONITOR       
========================================================================
Job:       #345 (agenticnav-replica-full) | Status: RUNNING
GPU:       NVIDIA L40S | 42°C, 135.0W | VRAM: 14.2 / 48.0 GB | Compute: 48%
Time:      Elapsed: 01h 15m / Limit: 360m
------------------------------------------------------------------------
Overall Progress: [============>                 ] 42.0% (126/300 runs)
Active Episode  : Repetition 2/3 | Episode 26/100
Estimated ETA   : ~02h 45m
------------------------------------------------------------------------
RUNNING REPLICATION METRICS:
  Success Rate (SR) : 74.6%  (Target: 76.0%)
  Path Weight (SPL) : 64.2%  (Target: 66.5%)
  Navigation Error  : 2.15m  (Target: 3.48m)
------------------------------------------------------------------------
RECENT FINISHED RUNS (Last 5):
  - Rep 2 Ep 025: [SUCCESS] SPL=0.710 | NE=1.12m | Category=success
  - Rep 2 Ep 024: [SUCCESS] SPL=0.680 | NE=1.85m | Category=success
  - Rep 2 Ep 023: [FAILED ] SPL=0.000 | NE=3.45m | Category=stop_premature
========================================================================
(Polling every 5s. Press Ctrl+C to stop watching)
```
*Note:* Pressing `Ctrl+C` exits the local terminal monitor without affecting the remote job.

---

### Step 5: Download Results & Trajectory Traces
Once the job finishes, download the complete metrics, summary, and trajectory traces to your local machine:
```powershell
python remote/download_results.py --remote /home/jovyan/scratch/agenticnav_paper_replica/runs/agenticnav-replica-full --local runs/remote_results
```

---

## 4. Complete Script Reference in `remote/`

| File | Purpose |
| :--- | :--- |
| [`config.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/config.py) | Cluster IP, API endpoints, authentication token, and SSL context helper. |
| [`status.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/status.py) | Real-time GPU stats, VRAM, system load, user budget, and queue state. |
| [`check_env.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/check_env.py) | Runs full environment, CUDA, Python, and storage diagnostics on GPU 1. |
| [`sync_all.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/sync_all.py) | Syncs entire codebase to `/home/jovyan/scratch/` and validates remote imports. |
| [`launch_benchmark_job.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/launch_benchmark_job.py) | Submits long-running benchmark jobs with watchdog heartbeat and auto-resumption. |
| [`track_progress.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/track_progress.py) | Real-time terminal progress dashboard with ASCII progress bar and running SR/SPL. |
| [`manage_job.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/manage_job.py) | Lists recent jobs, inspects logs, streams raw console output, or cancels jobs. |
| [`submit_job.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/submit_job.py) | Submits arbitrary custom batch scripts to the GPU queue. |
| [`download_results.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/download_results.py) | Bundles remote output directories and downloads them to your local PC. |
| [`connect_terminal.py`](file:///c:/Users/Kruti/Desktop/PROG/UAV/remote/connect_terminal.py) | Spawns an interactive bash terminal over WebSocket (when GPU 0 is free). |
