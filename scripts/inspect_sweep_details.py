import json
from collections import Counter
from pathlib import Path

# Run single pose or inspect why FF occurs
# Let's inspect the results in g64b_multiscene_sweep_results.json
sweep_file = Path("audit/interactive_session/g64b_multiscene_sweep_results.json")
with open(sweep_file) as f:
    data = json.load(f)

print(f"Elapsed time: {data['elapsed_seconds']}s")
print(f"Total actions: {data['total_actions']}")
print(f"Truth blocked: {data['truth_blocked']}, Truth free: {data['truth_free']}")

