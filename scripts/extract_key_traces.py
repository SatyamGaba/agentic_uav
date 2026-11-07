import json

def get_inst(t):
    inst = t.get("instruction")
    if isinstance(inst, dict):
        return inst.get("instruction_text", "")
    return str(inst)

def summarize_steps(steps):
    out = []
    for s in steps:
        step_idx = s.get("step_idx", s.get("node_idx"))
        action_type = s.get("action_type")
        payload = s.get("action_payload", {})
        post_ne = s.get("post_geodesic_distance_to_goal_m", s.get("geodesic_distance_to_goal_m"))
        outcome = s.get("outcome", {})
        turn_summary = []
        for t in s.get("turns", []):
            role = t.get("role")
            if role == "assistant":
                resp = t.get("response", {})
                tcs = resp.get("tool_calls", [])
                turn_summary.append(f"assistant: {[tc.get('name') for tc in tcs]}")
            elif role == "tool":
                c = str(t.get("content", ""))
                turn_summary.append(f"tool: {t.get('name')} -> {c[:60]}")
            elif role == "user" and t.get("arrival_verification"):
                turn_summary.append("user: [arrival_verification_prompt]")
        out.append({
            "step": step_idx,
            "action": action_type,
            "payload": {k: v for k, v in payload.items() if k != "reason"},
            "reason": payload.get("reason", "")[:80],
            "post_ne_m": round(post_ne, 2) if post_ne else None,
            "outcome": outcome,
            "dialogue_flow": turn_summary
        })
    return out

with open("audit/dev_arm_l_5ep/traces/evaluation_traces.jsonl") as f:
    traces_l = {t['episode_id']: t for t in (json.loads(l) for l in f)}

with open("audit/dev_arm_p_5ep/traces/evaluation_traces.jsonl") as f:
    traces_p = {t['episode_id']: t for t in (json.loads(l) for l in f)}

trace_best = {
    "episode_id": "2021",
    "arm": "Arm L (Local Aids)",
    "instruction": get_inst(traces_l["2021"]),
    "metrics": traces_l["2021"]["metrics"],
    "category": "success",
    "steps": summarize_steps(traces_l["2021"]["steps"])
}

trace_missed_stop = {
    "episode_id": "2022",
    "arm": "Arm L (Local Aids)",
    "instruction": get_inst(traces_l["2022"]),
    "metrics": traces_l["2022"]["metrics"],
    "category": "stop_after_reach (stopped at 3.05m)",
    "steps": summarize_steps(traces_l["2022"]["steps"])
}

trace_deadlock = {
    "episode_id": "259",
    "arm": "Arm P (Paper-Faithful N_th=1)",
    "instruction": get_inst(traces_p["259"]),
    "metrics": traces_p["259"]["metrics"],
    "category": "safety_deadlock (5 consecutive reselects on N_th=1)",
    "steps": summarize_steps(traces_p["259"]["steps"])
}

trace_violation = {
    "episode_id": "2021",
    "arm": "Arm P (Paper-Faithful N_th=1, no floor prompt)",
    "instruction": get_inst(traces_p["2021"]),
    "metrics": traces_p["2021"]["metrics"],
    "category": "protocol_violation",
    "steps": summarize_steps(traces_p["2021"]["steps"])
}

with open("audit/metric_validation/three_key_traces.json", "w") as f:
    json.dump({
        "best_episode_success": trace_best,
        "missed_stop_or_stop_after_reach": trace_missed_stop,
        "deadlock_episode_arm_p": trace_deadlock,
        "protocol_violation_arm_p": trace_violation
    }, f, indent=2)

print("Exported three_key_traces.json successfully!")
