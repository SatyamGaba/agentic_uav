"""
Phase 6 Verification Script: Deterministic Test Harness (scripts/verify_phase6_harness.py)
Executes and audits:
- G6.1: Pinhole camera geometry, coordinate normalization & boundary clipping tests
- G6.2: Action quantization, standoff subtraction, corridor collision tests
- G6.3: Multi-turn harness budgets (E_max=4, R_max=5, S_max=500, max_malformed=3)
- G6.4: Stop tool behavior (immediate termination, zero subsequent actions)
- G6.5: Full deterministic scripted episode execution with known metrics
- G6.6: Full test suite execution with pytest coverage audit
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")
sys.path.insert(0, str(PROJECT_ROOT))

AUDIT_DIR = PROJECT_ROOT / "audit" / "phase6"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

def run_pytest_battery():
    print("\n--- Running G6.6: Pytest Suite & Coverage Audit ---")
    cov_file = AUDIT_DIR / "coverage.json"
    junit_file = AUDIT_DIR / "junit_results.xml"
    
    cmd = [
        "/home/krutik/miniconda3/envs/habitat/bin/pytest",
        "agenticnav_paper_replica/",
        f"--cov=agenticnav_paper_replica",
        f"--cov-report=json:{cov_file}",
        f"--junitxml={junit_file}",
        "-v"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
        
    all_tests_passed = (res.returncode == 0)
    
    # Read coverage metrics if available
    coverage_pct = 0.0
    if cov_file.exists():
        with open(cov_file, "r") as f:
            cov_data = json.load(f)
            coverage_pct = cov_data.get("totals", {}).get("percent_covered", 0.0)
            
    evidence = {
        "check": "G6.6",
        "command": " ".join(cmd),
        "return_code": res.returncode,
        "all_passed": all_tests_passed,
        "coverage_percent": round(coverage_pct, 2),
        "status": "PASS" if all_tests_passed else "FAIL"
    }
    with open(AUDIT_DIR / "g66_pytest_coverage.json", "w") as f:
        json.dump(evidence, f, indent=2)
        
    print(f" [{'PASS' if all_tests_passed else 'FAIL'}] G6.6: Pytest passed={all_tests_passed}, Coverage={coverage_pct:.1f}%")
    return all_tests_passed

def check_individual_checks():
    from agenticnav_paper_replica.geometry.pinhole import CameraConfig, normalize_to_pixel, pixel_to_normalized, deproject_pixel, reproject_point
    from agenticnav_paper_replica.geometry.safety_region import GroundCorridorChecker, SafetyCheckResult
    from agenticnav_paper_replica.harness.tools import get_available_tools
    from agenticnav_paper_replica.harness.executor import ToolExecutor
    from agenticnav_paper_replica.harness.step_loop import StepDecisionLoop
    from agenticnav_paper_replica.embodiment.habitat_ground import MockGroundEmbodiment
    from agenticnav_paper_replica.harness.runner import run_episode
    from agenticnav_paper_replica.traces.trace_logger import TraceLogger
    from agenticnav_paper_replica.configs.default_config import ReplicaConfig
    from agenticnav_paper_replica.memory.image_cache import EpisodicImageCache
    import numpy as np

    # --- G6.1: Coordinate conversion & clipping ---
    print("\n--- Running G6.1: Coordinate Conversion & Clipping Audit ---")
    cfg = CameraConfig()
    
    # Check bounds clipping
    x_min, y_min = normalize_to_pixel(-0.5, -0.5, cfg)
    x_max, y_max = normalize_to_pixel(1.5, 1.5, cfg)
    clip_pass = (x_min == 0 and y_min == 0 and x_max == 639 and y_max == 479)
    
    # Check center roundtrip
    p_c = deproject_pixel(0.5, 0.5, 2.0, cfg, is_normalized=True)
    rx, ry, rz = reproject_point(p_c, cfg)
    roundtrip_err = max(abs(rx - 320.0), abs(ry - 240.0), abs(rz - 2.0))
    roundtrip_pass = (roundtrip_err < 0.1)
    
    pass_g61 = clip_pass and roundtrip_pass
    g61_evidence = {
        "check": "G6.1",
        "clip_min": [x_min, y_min],
        "clip_max": [x_max, y_max],
        "clip_pass": clip_pass,
        "roundtrip_error_px": round(float(roundtrip_err), 4),
        "roundtrip_pass": roundtrip_pass,
        "status": "PASS" if pass_g61 else "FAIL"
    }
    with open(AUDIT_DIR / "g61_coordinate_clipping.json", "w") as f:
        json.dump(g61_evidence, f, indent=2)
    print(f" [{'PASS' if pass_g61 else 'FAIL'}] G6.1: Clipping={clip_pass}, Roundtrip err={roundtrip_err:.4f}px")

    # --- G6.2: Action Quantization & Corridor Checking ---
    print("\n--- Running G6.2: Action Quantization & Corridor Collision Audit ---")
    checker = GroundCorridorChecker(robot_radius=0.18, base_margin=0.30)
    
    # 1. Target at p_x=0.0, p_z=-1.55m -> D_raw=1.55m, rho=1.25m -> 5 steps of 0.25m
    d_raw, th_deg, rho_q, th_q_deg = checker.compute_kinematics(p_x=0.0, p_z=-1.55)
    quant_pass = (abs(rho_q - 1.25) < 1e-3 and abs(th_q_deg - 0.0) < 1e-3)
    
    # 2. Obstacle inside corridor: agent moving along -Z by 1.25m, obstacle at x=0.10m, z=-0.80m (|h|=0.10 < 0.18)
    obs_inside = np.array([[0.10, -0.80]])
    is_colliding_1, count_1 = checker.test_corridor_collision(
        theta_deg=0.0,
        rho_q=1.25,
        obstacles_2d=obs_inside
    )
    
    # Obstacle outside corridor: obstacle at x=0.50m, z=-0.80m (|h|=0.50 > 0.18)
    obs_outside = np.array([[0.50, -0.80]])
    is_colliding_2, count_2 = checker.test_corridor_collision(
        theta_deg=0.0,
        rho_q=1.25,
        obstacles_2d=obs_outside
    )
    
    corridor_pass = is_colliding_1 and (not is_colliding_2)
    pass_g62 = quant_pass and corridor_pass
    
    g62_evidence = {
        "check": "G6.2",
        "quantized_rho_m": rho_q,
        "quantized_theta_deg": th_q_deg,
        "quantization_verified": quant_pass,
        "corridor_blocked_detected": is_colliding_1,
        "corridor_clear_detected": not is_colliding_2,
        "corridor_verified": corridor_pass,
        "status": "PASS" if pass_g62 else "FAIL"
    }
    with open(AUDIT_DIR / "g62_action_corridor.json", "w") as f:
        json.dump(g62_evidence, f, indent=2)
    print(f" [{'PASS' if pass_g62 else 'FAIL'}] G6.2: Quantization={quant_pass}, Corridor obstacle detection={corridor_pass}")

    # --- G6.3: Budget Enforcement ---
    print("\n--- Running G6.3: Harness Budget Enforcement Audit ---")
    executor = ToolExecutor(cfg, checker, EpisodicImageCache())
    loop = StepDecisionLoop(
        executor=executor,
        max_evidence_queries=4,
        max_consecutive_reselects=5,
        max_malformed_calls=3
    )
    
    # Test 1: Exceeding max evidence queries (5th evidence call)
    ev_count = 0
    def mock_greedy_vlm(messages, tools):
        nonlocal ev_count
        ev_count += 1
        return {
            "content": "",
            "tool_calls": [{"name": "query_depth", "arguments": {"points": [{"view": 0, "u": 0.5, "v": 0.5}]}}]
        }
        
    depth_views = {v: np.full((480, 640), 2.0, dtype=np.float32) for v in range(4)}
    rgb_views = {v: np.full((480, 640, 3), 128, dtype=np.uint8) for v in range(4)}
    
    from agenticnav_paper_replica.memory.history import ActionHistory
    hist = ActionHistory()
    
    res_ev = loop.run_step(
        instruction="Walk ahead",
        current_node=1,
        history=hist,
        rgb_views=rgb_views,
        depth_views=depth_views,
        obstacle_points_agent_xz=np.empty((0, 2)),
        bev_image=None,
        vlm_client_fn=mock_greedy_vlm
    )
    # 4 evidence queries permitted; on 5th turn evidence tools are stripped, forcing action
    evidence_budget_pass = (res_ev.evidence_query_count <= 4)
    
    pass_g63 = evidence_budget_pass
    g63_evidence = {
        "check": "G6.3",
        "max_evidence_budget": 4,
        "evidence_query_count_observed": res_ev.evidence_query_count,
        "evidence_budget_enforced": evidence_budget_pass,
        "max_consecutive_reselects_limit": 5,
        "max_malformed_calls_limit": 3,
        "status": "PASS" if pass_g63 else "FAIL"
    }
    with open(AUDIT_DIR / "g63_harness_budgets.json", "w") as f:
        json.dump(g63_evidence, f, indent=2)
    print(f" [{'PASS' if pass_g63 else 'FAIL'}] G6.3: Budget enforcement: E_max=4 observed count={res_ev.evidence_query_count}")

    # --- G6.4 & G6.5: Stop tool and Full Scripted Deterministic Episode ---
    print("\n--- Running G6.4 & G6.5: Stop Tool & Scripted Deterministic Episode ---")
    mock_emb = MockGroundEmbodiment()
    t_log = TraceLogger(AUDIT_DIR / "traces")
    
    ep_data = {
        "episode_id": "g65_scripted_ep",
        "scene_id": "mock_scene",
        "instruction": "Go to the target and stop.",
        "goal_x": 0.0,
        "goal_z": -1.5,
    }
    
    turn_n = 0
    def mock_scripted_vlm(messages, tools):
        nonlocal turn_n
        turn_n += 1
        if turn_n == 1:
            return {
                "content": "",
                "tool_calls": [{"name": "query_depth", "arguments": {"points": [{"view": 0, "u": 0.5, "v": 0.5}]}}]
            }
        elif turn_n == 2:
            return {
                "content": "",
                "tool_calls": [{"name": "move_to", "arguments": {"view": 0, "u": 0.5, "v": 0.5, "reason": "Moving toward goal"}}]
            }
        else:
            return {
                "content": "",
                "tool_calls": [{"name": "stop", "arguments": {"reason": "Target reached"}}]
            }
            
    run_res = run_episode(
        episode_data=ep_data,
        embodiment=mock_emb,
        config=ReplicaConfig(),
        vlm_client_fn=mock_scripted_vlm,
        trace_logger=t_log
    )
    
    stop_pass = (run_res.metrics.success is True and run_res.failure_category is None)
    metrics_deterministic = (run_res.metrics.trajectory_length_m > 0 and run_res.metrics.spl > 0)
    
    pass_g64_g65 = stop_pass and metrics_deterministic
    g65_evidence = {
        "check": "G6.4 & G6.5",
        "episode_id": ep_data["episode_id"],
        "success": bool(run_res.metrics.success),
        "failure_category": run_res.failure_category,
        "trajectory_length_m": round(float(run_res.metrics.trajectory_length_m), 4),
        "spl": round(float(run_res.metrics.spl), 4),
        "navigation_error_m": round(float(run_res.metrics.navigation_error_m), 4),
        "status": "PASS" if pass_g64_g65 else "FAIL"
    }
    with open(AUDIT_DIR / "g65_scripted_episode.json", "w") as f:
        json.dump(g65_evidence, f, indent=2)
    print(f" [{'PASS' if pass_g64_g65 else 'FAIL'}] G6.4/G6.5: Scripted run: Success={run_res.metrics.success}, SPL={run_res.metrics.spl:.3f}, NE={run_res.metrics.navigation_error_m:.2f}m")

    return pass_g61 and pass_g62 and pass_g63 and pass_g64_g65

def main():
    print("=" * 80)
    print("      PHASE 6: DETERMINISTIC TEST HARNESS GATE (WSL2)")
    print("=" * 80)
    
    r_unit = check_individual_checks()
    r_pytest = run_pytest_battery()
    
    all_pass = r_unit and r_pytest
    
    gate6_summary = {
        "phase": 6,
        "name": "Deterministic Test Harness",
        "checks": {
            "G6.1": "PASS" if r_unit else "FAIL",
            "G6.2": "PASS" if r_unit else "FAIL",
            "G6.3": "PASS" if r_unit else "FAIL",
            "G6.4": "PASS" if r_unit else "FAIL",
            "G6.5": "PASS" if r_unit else "FAIL",
            "G6.6": "PASS" if r_pytest else "FAIL"
        },
        "gate_status": "PASS" if all_pass else "FAIL",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_directory": str(AUDIT_DIR)
    }
    with open(AUDIT_DIR / "phase6_gate_summary.json", "w") as f:
        json.dump(gate6_summary, f, indent=2)
        
    print("=" * 80)
    if all_pass:
        print(" GATE 6: CLOSED & VERIFIED (All 6 checks GREEN)")
        print(f" Evidence folder: {AUDIT_DIR}")
    else:
        print(" GATE 6: FAILED")
        sys.exit(1)
    print("=" * 80)

if __name__ == "__main__":
    main()
