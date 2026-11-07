import json
from collections import Counter
from pathlib import Path

sweep_file = Path("audit/interactive_session/g64b_multiscene_sweep_results.json")
with open(sweep_file) as f:
    data = json.load(f)

print(f"Total actions: {data['total_actions']}")
print(f"Truth blocked: {data['truth_blocked']}, Truth free: {data['truth_free']}")

best = data["sweep_grid"][0]
print(f"\nBest config: r_a={best['r_a']}, y_min={best['y_min']}, y_max={best['y_max']}")
print(f"  FF={best['FF']}, FB={best['FB']}, TB={best['TB']}, TF={best['TF']}, Acc={best['accuracy']*100:.1f}%")

# Let's inspect the top 10 configs
print("\nTop 10 configs by accuracy:")
sorted_by_acc = sorted(data["sweep_grid"], key=lambda x: -x["accuracy"])
for c in sorted_by_acc[:10]:
    print(f"  r_a={c['r_a']:.2f}, y_min={c['y_min']:.2f}, y_max={c['y_max']:.2f} -> Acc={c['accuracy']*100:.2f}%, FF={c['FF']} ({c['collision_risk_ff_rate']*100:.1f}%), FB={c['FB']} ({c['deadlock_risk_fb_rate']*100:.1f}%)")

