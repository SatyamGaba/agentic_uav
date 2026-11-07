"""
Verify that all Phase 0 to Phase 10 gate summaries exist and have PASS status.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")
audit_dir = PROJECT_ROOT / "audit"

all_green = True
gate_results = {}

for phase in range(11):
    phase_dir = audit_dir / f"phase{phase}"
    summary_file = phase_dir / f"phase{phase}_gate_summary.json"
    if not summary_file.exists():
        print(f"Phase {phase}: MISSING ({summary_file})")
        all_green = False
        gate_results[f"Phase_{phase}"] = "MISSING"
        continue
    try:
        with open(summary_file) as f:
            data = json.load(f)
        status = data.get("status", "FAIL")
        print(f"Phase {phase}: {status}")
        gate_results[f"Phase_{phase}"] = status
        if status != "PASS":
            all_green = False
    except Exception as e:
        print(f"Phase {phase}: ERROR ({e})")
        all_green = False
        gate_results[f"Phase_{phase}"] = "ERROR"

print(f"\nAll gates status: {'ALL GREEN' if all_green else 'FAILURES DETECTED'}")

