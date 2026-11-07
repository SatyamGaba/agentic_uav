import json, glob
from pathlib import Path

ep_dir = Path("audit/dev_20_eval_500/arm_l/episodes")
ep_files = sorted(list(ep_dir.glob("*.json")))

results = []
for p in ep_files:
    with open(p) as f:
        d = json.load(f)
        if "metrics" in d:
            results.append(d)

print(f"Total valid episode results: {len(results)}/20")
sr_cnt = sum(1 for r in results if r["metrics"]["success"])
osr_cnt = sum(1 for r in results if r["metrics"]["oracle_success"])
deadlock_cnt = sum(1 for r in results if r["failure_category"] == "safety_deadlock")
premature_cnt = sum(1 for r in results if r["failure_category"] == "stop_premature")
budget_cnt = sum(1 for r in results if r["failure_category"] == "budget_exceeded")
protocol_cnt = sum(1 for r in results if r["failure_category"] == "protocol_violation")
zero_actions = sum(1 for r in results if r["metrics"]["simulator_action_count"] == 0)

mean_ne = sum(r["metrics"]["navigation_error_m"] for r in results) / len(results)
mean_actions = sum(r["metrics"]["simulator_action_count"] for r in results) / len(results)
mean_spl = sum(r["metrics"]["spl"] for r in results) / len(results)

summary_20 = {
    "num_episodes": len(results),
    "success_rate_pct": round(100.0 * sr_cnt / len(results), 2),
    "oracle_success_rate_pct": round(100.0 * osr_cnt / len(results), 2),
    "mean_spl_pct": round(100.0 * mean_spl, 2),
    "mean_navigation_error_m": round(mean_ne, 2),
    "mean_simulator_actions": round(mean_actions, 1),
    "deadlock_rate_pct": round(100.0 * deadlock_cnt / len(results), 2),
    "zero_action_rate_pct": round(100.0 * zero_actions / len(results), 2),
    "budget_exceeded_rate_pct": round(100.0 * budget_cnt / len(results), 2),
    "failure_taxonomy": {
        "success": sr_cnt,
        "safety_deadlock": deadlock_cnt,
        "stop_premature": premature_cnt,
        "budget_exceeded": budget_cnt,
        "protocol_violation": protocol_cnt,
    }
}

with open("audit/dev_20_eval_500/arm_l/benchmark_summary.json", "w") as f:
    json.dump(summary_20, f, indent=2)

print("\nOfficial 20-Episode Dev Summary (Arm L, 500 budget):")
print(json.dumps(summary_20, indent=2))
