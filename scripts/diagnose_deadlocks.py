#!/usr/bin/env python3
"""Detailed Diagnostic of the 12 safety_deadlock episodes in audit/phase9_postfix.

Categorizes failure causes:
1. Above-horizon targets (v < 0.5) / wall selection
2. Stair / floor clearance rejection (corridor blocked, no motion)
3. Target coordinate distribution (u, v) and views chosen
4. Reselect causes ('no_motion', 'corridor_blocked', etc.)
5. Whether the agent moved at all (zero-action vs non-zero action)
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

TRACES_FILE = Path("audit/phase9_postfix/traces/evaluation_traces.jsonl")
SUMMARY_FILE = Path("audit/phase9_postfix/benchmark_summary.json")

def main():
    with open(SUMMARY_FILE, "r") as f:
        summary = json.load(f)
    
    ep_results = {e["episode_id"]: e for e in summary.get("episodes", [])}
    
    with open(TRACES_FILE, "r") as f:
        traces = [json.loads(line) for line in f if line.strip()]
        
    print(f"Total traces loaded: {len(traces)}")
    
    deadlock_traces = [t for t in traces if ep_results.get(t["episode_id"], {}).get("failure_category") == "safety_deadlock"]
    print(f"Total safety_deadlock traces: {len(deadlock_traces)}")
    
    categories = {
        "zero_action_deadlock": [],
        "multi_action_deadlock": [],
    }
    
    v_coords = []
    u_coords = []
    views_called = Counter()
    reselect_causes = Counter()
    per_episode_diagnosis = []
    
    for t in deadlock_traces:
        ep_id = t["episode_id"]
        scene = t.get("scene_id", "")
        instr = t.get("instruction", "")
        steps = t.get("steps", [])
        m = ep_results.get(ep_id, {}).get("metrics", {})
        actions_taken = m.get("simulator_action_count", 0)
        
        is_zero_action = (actions_taken == 0 or len(steps) <= 1)
        if is_zero_action:
            categories["zero_action_deadlock"].append(ep_id)
        else:
            categories["multi_action_deadlock"].append(ep_id)
            
        # Collect tool calls across turns
        ep_reselect_causes = []
        ep_calls = []
        above_horizon_count = 0
        total_move_calls = 0
        
        for step in steps:
            for turn in step.get("turns", []):
                role = turn.get("role")
                if role == "assistant":
                    resp = turn.get("response", {})
                    tcalls = resp.get("tool_calls", [])
                    for tc in tcalls:
                        name = tc.get("name")
                        args = tc.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except:
                                args = {}
                        if name == "move_to":
                            total_move_calls += 1
                            v = args.get("v", 0.5)
                            u = args.get("u", 0.5)
                            view = args.get("view", 0)
                            v_coords.append(v)
                            u_coords.append(u)
                            views_called[view] += 1
                            if v < 0.50:
                                above_horizon_count += 1
                            ep_calls.append({"view": view, "u": u, "v": v, "reason": args.get("reason", "")[:60]})
                elif role == "tool":
                    name = turn.get("name")
                    content = turn.get("content", "")
                    if name == "move_to":
                        try:
                            payload = json.loads(content)
                            cause = payload.get("cause", "unknown")
                            reselect_causes[cause] += 1
                            ep_reselect_causes.append(cause)
                        except:
                            pass
                            
        per_episode_diagnosis.append({
            "episode_id": ep_id,
            "scene_id": scene,
            "instruction": instr,
            "actions_taken": actions_taken,
            "is_zero_action": is_zero_action,
            "total_move_calls": total_move_calls,
            "above_horizon_count": above_horizon_count,
            "above_horizon_ratio": above_horizon_count / max(1, total_move_calls),
            "reselect_causes": Counter(ep_reselect_causes),
            "sample_calls": ep_calls[:4],
        })
        
    print("\n=======================================================")
    print(f"ZERO-ACTION VS MULTI-ACTION DEADLOCKS:")
    print(f"  Zero-Action Deadlocks: {len(categories['zero_action_deadlock'])} / 12 ({len(categories['zero_action_deadlock'])/12*100:.1f}%)")
    print(f"  Multi-Action Deadlocks: {len(categories['multi_action_deadlock'])} / 12 ({len(categories['multi_action_deadlock'])/12*100:.1f}%)")
    print("=======================================================")
    
    print("\nRESELECT CAUSES BREAKDOWN:")
    for cause, count in reselect_causes.most_common():
        print(f"  {cause}: {count} occurrences ({count/sum(reselect_causes.values())*100:.1f}%)")
        
    print("\nTARGET COORDINATE (u, v) STATISTICS:")
    if v_coords:
        import numpy as np
        v_arr = np.array(v_coords)
        u_arr = np.array(u_coords)
        print(f"  v (vertical): mean={v_arr.mean():.3f}, std={v_arr.std():.3f}, min={v_arr.min():.3f}, max={v_arr.max():.3f}")
        print(f"  u (horizontal): mean={u_arr.mean():.3f}, std={u_arr.std():.3f}, min={u_arr.min():.3f}, max={u_arr.max():.3f}")
        above_horiz_total = (v_arr < 0.50).sum()
        print(f"  Above-Horizon Calls (v < 0.50): {above_horiz_total} / {len(v_arr)} ({above_horiz_total/len(v_arr)*100:.1f}%)")
        print(f"  Views Selected: {dict(views_called)}")
        
    print("\nPER-EPISODE BREAKDOWN:")
    for d in per_episode_diagnosis:
        print(f"\n--- {d['episode_id']} ({d['scene_id']}) ---")
        print(f"  Instruction: \"{d['instruction'][:80]}...\"")
        print(f"  Actions taken: {d['actions_taken']} (Zero-action: {d['is_zero_action']})")
        print(f"  Above-horizon move calls: {d['above_horizon_count']} / {d['total_move_calls']} ({d['above_horizon_ratio']*100:.1f}%)")
        print(f"  Reselect causes: {dict(d['reselect_causes'])}")
        for sc in d["sample_calls"]:
            print(f"    Call: view={sc['view']}, u={sc['u']:.2f}, v={sc['v']:.2f} | Reason: {sc['reason']}")
            
    # Save diagnostic summary
    out_path = Path("audit/phase9_postfix/deadlock_diagnosis.json")
    with open(out_path, "w") as f:
        json.dump({
            "total_deadlocks": len(deadlock_traces),
            "zero_action_count": len(categories["zero_action_deadlock"]),
            "multi_action_count": len(categories["multi_action_deadlock"]),
            "reselect_causes": dict(reselect_causes),
            "above_horizon_ratio": float(above_horiz_total / len(v_coords)) if v_coords else 0.0,
            "episodes": per_episode_diagnosis,
        }, f, indent=2)
    print(f"\nDiagnostic evidence saved to {out_path}")

if __name__ == "__main__":
    main()

