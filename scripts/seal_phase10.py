"""
Seal Phase 10 and verify all gates Phase 0 - Phase 10.
"""

import sys
import json
import time
from pathlib import Path

PROJECT_ROOT = Path("/mnt/c/Users/Kruti/Desktop/PROG/UAV")
AUDIT_DIR = PROJECT_ROOT / "audit"

smoke_summary_path = AUDIT_DIR / "phase10" / "smoke_summary.json"
rep_report_path = AUDIT_DIR / "replication_report.md"

if not smoke_summary_path.exists():
    print("Smoke summary not found yet.")
    sys.exit(1)

with open(smoke_summary_path) as f:
    smoke_res = json.load(f)

pass_g10_1 = (smoke_res.get("status") == "PASS")
pass_g10_3 = rep_report_path.exists() and rep_report_path.stat().st_size > 1000

def is_phase_passed(data):
    if data.get("status") == "PASS":
        return True
    if data.get("gate_closed") is True:
        return True
    checks = data.get("checks", {})
    if checks and all((v == "PASS" or (isinstance(v, dict) and v.get("status") == "PASS")) for v in checks.values()):
        return True
    return False

# Check G10.2: All audit/phaseN/ gate summaries are green
all_gates_green = True
gate_checks = {}
for p in range(10):  # Phases 0 to 9
    p_path = AUDIT_DIR / f"phase{p}" / f"phase{p}_gate_summary.json"
    if not p_path.exists():
        all_gates_green = False
        gate_checks[f"phase_{p}"] = "MISSING"
        continue
    with open(p_path) as pf:
        p_data = json.load(pf)
    
    passed = is_phase_passed(p_data)
    gate_checks[f"phase_{p}"] = "PASS" if passed else "FAIL"
    if not passed:
        all_gates_green = False

pass_g10_2 = all_gates_green

all_green = pass_g10_1 and pass_g10_2 and pass_g10_3

phase10_summary = {
    "gate": "GATE 10",
    "phase": 10,
    "title": "Config Freeze and Regression Archive",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "status": "PASS" if all_green else "FAIL",
    "checks": {
        "G10.1_regression_smoke": {"status": "PASS" if pass_g10_1 else "FAIL", "path": str(smoke_summary_path)},
        "G10.2_all_phases_green": {"status": "PASS" if pass_g10_2 else "FAIL", "phase_matrix": gate_checks},
        "G10.3_replication_report": {"status": "PASS" if pass_g10_3 else "FAIL", "path": str(rep_report_path), "bytes": rep_report_path.stat().st_size},
    }
}

out_path = AUDIT_DIR / "phase10" / "phase10_gate_summary.json"
with open(out_path, "w") as f:
    json.dump(phase10_summary, f, indent=2)

print("\n" + "=" * 80)
print(f"GATE 10 STATUS: {phase10_summary['status']}")
print(f"All Phases Matrix:")
for p, s in gate_checks.items():
    print(f"  {p}: {s}")
print(f"Summary written to: {out_path}")
print("=" * 80)

if not all_green:
    sys.exit(1)
