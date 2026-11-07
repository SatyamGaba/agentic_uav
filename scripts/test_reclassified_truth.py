import json
from collections import Counter
from pathlib import Path

# Load all 2100 evaluated actions from g64b_multiscene_sweep_results.json
sweep_file = Path("audit/interactive_session/g64b_multiscene_sweep_results.json")
with open(sweep_file) as f:
    data = json.load(f)

print("Loaded sweep results.")

