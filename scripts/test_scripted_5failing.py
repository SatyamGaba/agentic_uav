import json
from pathlib import Path
from test_scripted_oracle_harness import test_scripted_oracle
import habitat_sim

# Run the 5 episodes
from scripts.test_scripted_oracle_harness import PROJECT_ROOT
dev_path = PROJECT_ROOT / "agenticnav_paper_replica/data/r2rce_dev_20.json"
with open(dev_path, "r", encoding="utf-8") as f:
    data = json.load(f)
episodes = data.get("episodes", data)

failing_ids = {"8628", "1868", "4164", "8627", "3819"}
filtered_eps = [e for e in episodes if str(e["episode_id"]) in failing_ids]
print(f"Testing {len(filtered_eps)} episodes: {[e['episode_id'] for e in filtered_eps]}")

# Run with test_scripted_oracle logic directly

