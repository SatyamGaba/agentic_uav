# MultiUAV-Plat / Agent4Drone Standalone Benchmark Environment

Welcome to the standalone **MultiUAV-Plat** platform environment designed specifically for VLM prompt testing, sub-goal planning, visual servoing, and closed-loop flight benchmarking.

---

## 📁 Directory Structure

```
multiauav_plat/
├── worlds/
│   ├── multiauav_plat.wbt          # Webots 3D Benchmark World (Manor, Tesla, Windmill, Forest)
│   └── .multiauav_plat.wbproj      # Webots Viewpoint Configuration
└── controllers/
    └── multiauav_plat_controller/
        ├── multiauav_plat_controller.py  # Main Supervisor Controller (Port 8081)
        ├── perception.py            # VLM Client, Target Memory Tracker & Verification
        ├── planning.py              # Mission Planner & Multi-Phase Sub-Goal State Machine
        ├── config.py                # Flight Constants & Telemetry Parameters
        ├── utils.py                 # Math & Geometry Helpers
        ├── server.py                # Dashboard HTTP Server (Port 8081)
        ├── index.html               # Real-time Web Control & Telemetry Dashboard
        └── test_prompts.py          # Benchmark Prompt Execution Runner
```

---

## 🚀 How to Run & Test with VLM and Prompts

### 1. Open the World in Webots
1. In Webots, select **File -> Open World...**
2. Navigate to: `c:\Users\Kruti\Desktop\PROG\UAV\multiauav_plat\worlds\multiauav_plat.wbt`
3. Click **Open**. Webots will load the environment and automatically launch `multiauav_plat_controller`.

### 2. Access the MultiUAV-Plat Web Dashboard
Open your web browser and go to:
[http://localhost:8081](http://localhost:8081)

From the dashboard you can:
- Dispatch mission prompts (e.g. `"move towards the manor and focus on its top left window"`).
- View live camera feeds and telemetry metrics in real-time.
- Monitor 6-Stage reasoning logs and spatial target beliefs.

### 3. Run Benchmark Prompt Tests Standalone
To test prompt decomposition directly against your local LM Studio / Qwen VLM model:
```bash
python multiauav_plat/controllers/multiauav_plat_controller/test_prompts.py
```
