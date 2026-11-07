# Checkpoint: Iteration 6 Zero-Shot Dual-Rate Cognitive UAV Architecture

**Date Saved:** October 29, 2026  
**Status:** Preserved and Checkpointed for Future Resumption  
**Primary Artifacts:**
- Core Architecture: `controllers/mavic_vlm_controller/`
  - System 1 Reflex Loop (10 Hz): `reflex/depth_regulator.py`, `reflex/safety_fsm.py`, `reflex/continuous_velocity.py`
  - System 2 Deliberative Engine (~0.02 Hz): `cognitive/intent_arbiter.py`, `cognitive/vlm_client.py`, `core/scene_graph_3d.py`
  - Master Coordinator: `main_controller.py` (`DualRateMasterController`)
  - Simulation Adapter: `adapters/airsim_adapter.py`
- Test Suite: 72 / 72 unit tests passing (`pytest controllers/mavic_vlm_controller/tests/`)
- Remote Hardware: NVIDIA L40S GPU (48GB VRAM) at `https://172.24.16.175` (Ananta GPU Hub)
- Unreal Engine Environment: `env_airsim_18` (`/home/jovyan/scratch/OpenFly-Platform/envs/airsim/env_airsim_18`)
- Empirical Flight Traces & Discovery Log: `EMPIRICAL_FLIGHT_DISCOVERY_LOG.md` (Defects #1 through #8 cataloged, including the 0.17m camera offset fix and sky sentinel filtering).

---

## Key Achievements Prior to Pause
1. **Physical Mount Fix**: Calibrated camera mount offset $(+0.25, 0.0, -0.12)\text{m}$ in `airsim_adapter.py` to prevent drone chassis self-occlusion.
2. **Sky Sentinel Filter**: Replaced $>100\text{m}$ sentinel values with $100.0\text{m}$ in `depth_regulator.py` to prevent false positive emergency climbs when facing open sky.
3. **Decoupled Asynchronous Scheduling**: Verified System 1 runs at continuous 10 Hz station-keeping hover while System 2 deliberates with uncapped tokens (~45–60s) via `asyncio.create_task()`.
4. **Task Decomposition**: First live DAG task decomposition succeeded on live Unreal Engine pixels (`[IntentArbiter] Decomposed instruction into 2 stages`).

---

## Instructions for Resuming Later
To resume the Dual-Rate AirSim flight runs:
1. Ensure FortiClient VPN is connected (`10.10.50.1`).
2. Run `python scratch/run_closed_loop_episode.py` with `timeout_s = 180.0` and `max_ticks = 250`.
3. Check `EMPIRICAL_FLIGHT_DISCOVERY_LOG.md` for telemetry benchmarks.

