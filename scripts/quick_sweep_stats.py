import json
from collections import Counter
from pathlib import Path

# Load results from g64b_multiscene_sweep_results.json
sweep_file = Path("audit/interactive_session/g64b_multiscene_sweep_results.json")
with open(sweep_file) as f:
    data = json.load(f)

print(f"Total actions: {data['total_actions']}")
print(f"Truth blocked: {data['truth_blocked']}, Truth free: {data['truth_free']}")

best = data["sweep_grid"][0]
print(f"\nTop Config: r_a={best['r_a']}, y_min={best['y_min']}, y_max={best['y_max']}")
print(f"  FF={best['FF']} ({best['collision_risk_ff_rate']*100:.1f}%), FB={best['FB']} ({best['deadlock_risk_fb_rate']*100:.1f}%), Acc={best['accuracy']*100:.1f}%")

