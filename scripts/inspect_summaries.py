import json
from pathlib import Path

for p in range(10):
    f = Path(f"/mnt/c/Users/Kruti/Desktop/PROG/UAV/audit/phase{p}/phase{p}_gate_summary.json")
    if f.exists():
        d = json.load(open(f))
        stat = d.get("status")
        gc = d.get("gate_closed")
        checks = d.get("checks", {})
        print(f"Phase {p}: status={stat}, gate_closed={gc}, checks={checks if isinstance(checks, dict) else len(checks)}")

