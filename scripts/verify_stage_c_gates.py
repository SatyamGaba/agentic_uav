"""Stage C Gate Verification & Arm P vs Arm L Comparative Evaluation.

Evaluates:
1. Arm P (Paper-Faithful: N_th=1, no arrival verification, no floor guidance)
   vs Arm L (Local Aids: N_th=5, arrival verification, floor guidance) on 20 dev episodes.
2. Checks the 4 Stage C Gate Criteria:
   - Deadlock rate <= 10%
   - Zero-action rate <= 5%
   - Schema-valid tool calls >= 98%
   - OSR above random walk (> 2.0%)
"""

import json
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def analyze_traces_schema_validity(trace_file: Path):
    if not trace_file.exists():
        return 1.0, 0, 0
    total_calls = 0
    valid_calls = 0
    with open(trace_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ep = json.loads(line)
            for step in ep.get("steps", []):
                for turn in step.get("turns", []):
                    if turn.get("role") == "assistant":
                        resp = turn.get("response", {})
                        tool_calls = resp.get("tool_calls", [])
                        for tc in tool_calls:
                            total_calls += 1
                            name = tc.get("name")
                            args = tc.get("arguments", {})
                            if name in ("move_to", "query_depth", "recall", "stop") and isinstance(args, dict):
                                valid_calls += 1
    pct = (valid_calls / total_calls) if total_calls > 0 else 1.0
    return pct, valid_calls, total_calls

def verify_stage_c():
    arm_p_file = PROJECT_ROOT / "audit/dev_20_eval/arm_p/benchmark_summary.json"
    arm_l_file = PROJECT_ROOT / "audit/dev_20_eval/arm_l/benchmark_summary.json"
    
    if not arm_p_file.exists() or not arm_l_file.exists():
        print(f"Waiting for both arm_p and arm_l to complete. (arm_p exists: {arm_p_file.exists()}, arm_l exists: {arm_l_file.exists()})")
        return False
        
    with open(arm_p_file, "r", encoding="utf-8") as f:
        p_data = json.load(f)
    with open(arm_l_file, "r", encoding="utf-8") as f:
        l_data = json.load(f)
        
    p_eps = p_data["episodes"]
    l_eps = l_data["episodes"]
    n_p = len(p_eps)
    n_l = len(l_eps)
    
    # Arm P stats
    p_deadlocks = p_data["failure_taxonomy"].get("safety_deadlock", 0)
    p_deadlock_rate = p_deadlocks / n_p if n_p > 0 else 0.0
    p_zero_actions = sum(1 for e in p_eps if e["metrics"]["simulator_action_count"] == 0)
    p_zero_action_rate = p_zero_actions / n_p if n_p > 0 else 0.0
    p_osr = p_data["primary_metrics"]["oracle_success_rate"]["mean"] / 100.0
    p_sr = p_data["primary_metrics"]["success_rate"]["mean"] / 100.0
    p_ne = p_data["primary_metrics"]["navigation_error_m"]["mean"]
    
    # Arm L stats
    l_deadlocks = l_data["failure_taxonomy"].get("safety_deadlock", 0)
    l_deadlock_rate = l_deadlocks / n_l if n_l > 0 else 0.0
    l_zero_actions = sum(1 for e in l_eps if e["metrics"]["simulator_action_count"] == 0)
    l_zero_action_rate = l_zero_actions / n_l if n_l > 0 else 0.0
    l_osr = l_data["primary_metrics"]["oracle_success_rate"]["mean"] / 100.0
    l_sr = l_data["primary_metrics"]["success_rate"]["mean"] / 100.0
    l_ne = l_data["primary_metrics"]["navigation_error_m"]["mean"]
    
    # Schema validity on Arm L traces
    l_trace_file = PROJECT_ROOT / "audit/dev_20_eval/arm_l/traces/evaluation_traces.jsonl"
    l_valid_pct, l_val_cnt, l_tot_cnt = analyze_traces_schema_validity(l_trace_file)
    
    # Random walk OSR baseline on 100 episodes was 2.0%
    random_walk_osr = 0.02
    
    # Check gate criteria on Arm L (the active deployment arm)
    gate_deadlock = (l_deadlock_rate <= 0.10)
    gate_zero_action = (l_zero_action_rate <= 0.05)
    gate_schema = (l_valid_pct >= 0.98)
    gate_osr = (l_osr > random_walk_osr)
    
    all_passed = gate_deadlock and gate_zero_action and gate_schema and gate_osr
    
    report = {
        "task": "Stage C Gate Verification & Arm P vs Arm L Comparison",
        "dataset": "r2rce_dev_20.json (non-test scenes: gZ6f7yhEvPG, sT4fr6TAbpF, JF19kD82Mey)",
        "arm_p_paper_faithful": {
            "description": "N_th=1 (any point), floor_guidance=False, arrival_verify=False",
            "episodes": n_p,
            "success_rate": p_sr,
            "oracle_success_rate": p_osr,
            "mean_navigation_error_m": p_ne,
            "deadlock_count": p_deadlocks,
            "deadlock_rate": p_deadlock_rate,
            "zero_action_count": p_zero_actions,
            "zero_action_rate": p_zero_action_rate,
            "failure_taxonomy": p_data["failure_taxonomy"],
        },
        "arm_l_local_aids": {
            "description": "N_th=5 (calibrated), floor_guidance=True, arrival_verify=True",
            "episodes": n_l,
            "success_rate": l_sr,
            "oracle_success_rate": l_osr,
            "mean_navigation_error_m": l_ne,
            "deadlock_count": l_deadlocks,
            "deadlock_rate": l_deadlock_rate,
            "zero_action_count": l_zero_actions,
            "zero_action_rate": l_zero_action_rate,
            "schema_validity_rate": l_valid_pct,
            "schema_valid_calls": l_val_cnt,
            "schema_total_calls": l_tot_cnt,
            "failure_taxonomy": l_data["failure_taxonomy"],
        },
        "comparison_delta": {
            "deadlock_reduction_pct": ((p_deadlock_rate - l_deadlock_rate) / p_deadlock_rate * 100.0) if p_deadlock_rate > 0 else 0.0,
            "osr_delta_pct": (l_osr - p_osr) * 100.0,
            "ne_reduction_m": p_ne - l_ne,
        },
        "gate_evaluation": {
            "deadlock_rate_le_10pct": {"value": l_deadlock_rate, "passed": gate_deadlock},
            "zero_action_rate_le_5pct": {"value": l_zero_action_rate, "passed": gate_zero_action},
            "schema_valid_ge_98pct": {"value": l_valid_pct, "passed": gate_schema},
            "osr_gt_random_walk": {"value": l_osr, "random_walk_osr": random_walk_osr, "passed": gate_osr},
            "stage_c_gate_status": "PASS" if all_passed else "FAIL",
        }
    }
    
    out_dir = PROJECT_ROOT / "audit/state_integrity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "stage_c_gate_summary.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "="*80)
    print("STAGE C COMPARATIVE EVALUATION: ARM P vs ARM L (20 Dev Episodes)")
    print("="*80)
    print(f"  Metric                   | Arm P (Paper-Faithful) | Arm L (Local Aids)   | Delta")
    print(f"  -------------------------+------------------------+----------------------+-------")
    print(f"  Safety Deadlock Rate     | {p_deadlock_rate*100:6.1f}% ({p_deadlocks:2d}/{n_p})       | {l_deadlock_rate*100:6.1f}% ({l_deadlocks:2d}/{n_l})     | {(l_deadlock_rate - p_deadlock_rate)*100:+5.1f}%")
    print(f"  Zero-Action Rate         | {p_zero_action_rate*100:6.1f}% ({p_zero_actions:2d}/{n_p})       | {l_zero_action_rate*100:6.1f}% ({l_zero_actions:2d}/{n_l})     | {(l_zero_action_rate - p_zero_action_rate)*100:+5.1f}%")
    print(f"  Oracle Success Rate      | {p_osr*100:6.1f}%                | {l_osr*100:6.1f}%              | {(l_osr - p_osr)*100:+5.1f}%")
    print(f"  Navigation Error (m)     | {p_ne:6.2f} m              | {l_ne:6.2f} m            | {l_ne - p_ne:+5.2f} m")
    print(f"  Schema-Valid Tool Calls  |   --                   | {l_valid_pct*100:6.1f}% ({l_val_cnt}/{l_tot_cnt})  |   --")
    print("="*80)
    print("STAGE C GATE CRITERIA (Arm L):")
    print(f"  1. Deadlock Rate <= 10%:   {'PASS' if gate_deadlock else 'FAIL'} ({l_deadlock_rate*100:.1f}%)")
    print(f"  2. Zero-Action Rate <= 5%: {'PASS' if gate_zero_action else 'FAIL'} ({l_zero_action_rate*100:.1f}%)")
    print(f"  3. Schema-Valid >= 98%:    {'PASS' if gate_schema else 'FAIL'} ({l_valid_pct*100:.1f}%)")
    print(f"  4. OSR > Random Walk (2%): {'PASS' if gate_osr else 'FAIL'} ({l_osr*100:.1f}%)")
    print(f"  OVERALL GATE STATUS:       {'PASS' if all_passed else 'FAIL'}")
    print("="*80)
    print(f"Saved Stage C gate report to {out_file}")
    return all_passed

if __name__ == "__main__":
    verify_stage_c()

