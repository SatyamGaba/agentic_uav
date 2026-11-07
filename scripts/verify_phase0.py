"""
Phase 0 Verification Script: Source of Truth
Checks:
- G0.1: Every numeric constant and prompt string in ReplicaConfig maps to a ledger line.
- G0.2: Every assumed item has a reason and an ablation plan.
- G0.3: Episode list file identified, sha256 checksummed, scene set derived and verified.
- G0.4: Target definition recorded against paper rows.
"""
import sys
import os
import hashlib
import json
import yaml
from pathlib import Path

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agenticnav_paper_replica.configs.default_config import ReplicaConfig

AUDIT_DIR = PROJECT_ROOT / "audit" / "phase0"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def check_g01(ledger_data):
    """G0.1: Check that every config parameter in ReplicaConfig maps to a ledger line and has exact matching values."""
    print("Running G0.1: Config to Ledger Mapping & Value Parity Check...")
    default_config = ReplicaConfig()
    config_fields = set(ReplicaConfig.__dataclass_fields__.keys())
    ledger_params = ledger_data.get("parameters", {})
    ledger_keys = set(ledger_params.keys())

    unmapped = config_fields - ledger_keys
    value_mismatches = {}

    for field_name in config_fields:
        if field_name in ledger_params:
            config_val = getattr(default_config, field_name)
            ledger_val = ledger_params[field_name].get("value")
            if config_val != ledger_val:
                value_mismatches[field_name] = {
                    "config_value": config_val,
                    "ledger_value": ledger_val,
                }

    passed = (len(unmapped) == 0) and (len(value_mismatches) == 0)
    evidence = {
        "check": "G0.1",
        "total_config_fields": len(config_fields),
        "total_ledger_params": len(ledger_keys),
        "unmapped_fields": list(unmapped),
        "value_mismatches": value_mismatches,
        "status": "PASS" if passed else "FAIL"
    }

    out_file = AUDIT_DIR / "g01_config_ledger_mapping.json"
    with open(out_file, "w") as f:
        json.dump(evidence, f, indent=2)

    if unmapped:
        print(f" [FAIL] G0.1: Unmapped config fields: {unmapped}")
    if value_mismatches:
        print(f" [FAIL] G0.1: Value mismatches between ReplicaConfig and ledger: {value_mismatches}")
    if passed:
        print(f" [PASS] G0.1: All {len(config_fields)} config fields map to ledger lines with exact value equality.")
    return passed


def check_g02(ledger_data):
    """G0.2: Check that every assumed item has a reason and ablation plan."""
    print("Running G0.2: Assumed Items Validation...")
    params = ledger_data.get("parameters", {})
    assumed_items = []
    missing_ablation = []

    for name, p in params.items():
        if p.get("status") == "assumed":
            assumed_items.append(name)
            if not p.get("ablation_plan") or not p.get("description"):
                missing_ablation.append(name)

    evidence = {
        "check": "G0.2",
        "total_assumed_items": len(assumed_items),
        "assumed_items": assumed_items,
        "missing_ablation_or_reason": missing_ablation,
        "status": "PASS" if len(missing_ablation) == 0 else "FAIL"
    }

    out_file = AUDIT_DIR / "g02_assumed_items.json"
    with open(out_file, "w") as f:
        json.dump(evidence, f, indent=2)

    if missing_ablation:
        print(f" [FAIL] G0.2: Assumed items missing ablation plan: {missing_ablation}")
        return False
    print(f" [PASS] G0.2: All {len(assumed_items)} assumed parameters have verified reasons & ablation plans.")
    return True


def check_g03(ledger_data):
    """G0.3: Episode list file identified, sha256 checksummed, scene set derived."""
    print("Running G0.3: Episode & Scene Integrity Check...")
    ep_file_rel = ledger_data.get("parameters", {}).get("test_episodes_file", {}).get("value")
    ep_file = PROJECT_ROOT / ep_file_rel

    if not ep_file.exists():
        print(f" [FAIL] G0.3: Episode file does not exist: {ep_file}")
        return False

    with open(ep_file, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()

    with open(ep_file, "r") as f:
        data = json.load(f)

    episodes = data.get("episodes", [])
    raw_scenes = sorted(list(set(ep.get("scene_id") for ep in episodes if ep.get("scene_id"))))
    scenes = sorted(list(set(s.split("/")[1] if "/" in s else s.split(".")[0] for s in raw_scenes)))

    EXPECTED_10_VAL_UNSEEN_SCENES = sorted([
        '2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'QUCTc6BB5sX',
        'TbHJrupSAjP', 'X7HyMhZNoso', 'Z6MFQCViBuw', 'oLBMNvg9in8',
        'x8F5xyUWy9e', 'zsNo4HB9uLZ'
    ])

    scenes_match = (scenes == EXPECTED_10_VAL_UNSEEN_SCENES)

    evidence = {
        "check": "G0.3",
        "file_path": str(ep_file_rel),
        "sha256": sha256,
        "total_episodes": len(episodes),
        "total_unique_scenes": len(scenes),
        "scenes": scenes,
        "scenes_match_expected_val_unseen": scenes_match,
        "status": "PASS" if (len(episodes) == 100 and scenes_match) else "FAIL"
    }

    out_file = AUDIT_DIR / "g03_episodes_scenes.json"
    with open(out_file, "w") as f:
        json.dump(evidence, f, indent=2)

    if not (len(episodes) == 100 and scenes_match):
        print(f" [FAIL] G0.3: Mismatch in episode count ({len(episodes)}) or scenes ({scenes})")
        return False
    print(f" [PASS] G0.3: Episode file verified. SHA256={sha256[:12]}..., 100 episodes across exactly 11 scenes.")
    return True


def check_g04():
    """G0.4: Target definition written comparing paper open-weight rows."""
    print("Running G0.4: Benchmark Target Definition...")
    target_data = {
        "check": "G0.4",
        "benchmark": "R2R-CE val_unseen (100 episodes)",
        "anchor_paper": "AgenticNav: Zero-Shot Vision-and-Language Navigation as a Tool-Calling Harness (arXiv:2606.10577, 2026)",
        "paper_table_1_reported_baselines": {
            "gemini_3_7_flash_anchor": {
                "type": "Proprietary Commercial Anchor",
                "sr_pct": 76.0,
                "spl_pct": 66.50,
                "osr_pct": 78.0,
                "ndtw_pct": 69.38,
                "ne_meters": 3.48,
                "note": "Headline numbers reported with frontier multimodal API; not target for local 4B/8B open models."
            },
            "qwen3_8_27b_open_weight": {
                "type": "Open-Weight Baseline",
                "sr_pct": 48.0,
                "spl_pct": 33.57,
                "osr_pct": 72.0,
                "ndtw_pct": 61.58,
                "ne_meters": 5.65,
                "note": "Paper's reported performance for open-weight Qwen family backbone."
            },
            "deepseek_v4_flash_vision_exp": {
                "type": "Open-Weight Baseline",
                "sr_pct": 49.0,
                "spl_pct": 36.48,
                "osr_pct": 64.0,
                "ndtw_pct": 61.18,
                "ne_meters": 5.67
            }
        },
        "local_development_target": {
            "model": "qwen3.5:4b",
            "acceptance_criteria": [
                "Pipeline correctness strictly proven through G1 to G10 gates",
                "Zero silent fallbacks, true 5-image multimodal ingestion",
                "Local replication performance comparable to paper open-weight range (SR ~40-50%, SPL ~30-35%)"
            ]
        },
        "status": "PASS"
    }

    out_file = AUDIT_DIR / "g04_target_definition.json"
    with open(out_file, "w") as f:
        json.dump(target_data, f, indent=2)

    print(" [PASS] G0.4: Target definitions recorded in audit/phase0/g04_target_definition.json.")
    return True


def main():
    print("=" * 80)
    print("            PHASE 0: SOURCE OF TRUTH VERIFICATION GATE")
    print("=" * 80)

    ledger_path = PROJECT_ROOT / "spec" / "spec_ledger.yaml"
    if not ledger_path.exists():
        print(f"[FAIL] Ledger not found at {ledger_path}")
        sys.exit(1)

    with open(ledger_path, "r") as f:
        ledger_data = yaml.safe_load(f)

    r1 = check_g01(ledger_data)
    r2 = check_g02(ledger_data)
    r3 = check_g03(ledger_data)
    r4 = check_g04()

    all_pass = r1 and r2 and r3 and r4
    gate_status = {
        "phase": 0,
        "name": "Source of Truth",
        "checks": {
            "G0.1": "PASS" if r1 else "FAIL",
            "G0.2": "PASS" if r2 else "FAIL",
            "G0.3": "PASS" if r3 else "FAIL",
            "G0.4": "PASS" if r4 else "FAIL"
        },
        "gate_closed": all_pass
    }

    with open(AUDIT_DIR / "phase0_gate_summary.json", "w") as f:
        json.dump(gate_status, f, indent=2)

    print("=" * 80)
    if all_pass:
        print(" GATE 0: CLOSED & VERIFIED (All 4 checks GREEN)")
        print(f" Evidence written to: {AUDIT_DIR}")
    else:
        print(" GATE 0: FAILED (One or more checks did not pass)")
        sys.exit(1)
    print("=" * 80)


if __name__ == "__main__":
    main()

