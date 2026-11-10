import base64
import gzip
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("scripts").resolve()))
import cluster_client

files = [
    "agenticnav_paper_replica/geometry/pinhole.py",
    "agenticnav_paper_replica/harness/executor.py",
    "agenticnav_paper_replica/harness/prompts.py",
    "agenticnav_paper_replica/harness/runner.py",
    "agenticnav_paper_replica/harness/step_loop.py",
    "agenticnav_paper_replica/harness/tests/test_remediation_capabilities.py",
]

print("Generating compressed git patch for remediation changes (f2ac482 -> HEAD)...")
patch_bytes = subprocess.check_output(["git", "diff", "f2ac482", "HEAD", "--"] + files)
gz_bytes = gzip.compress(patch_bytes)
patch_b64 = base64.b64encode(gz_bytes).decode("ascii")
print(f"Patch ready: raw={len(patch_bytes)}B, gzip={len(gz_bytes)}B, b64={len(patch_b64)} chars.")

hf_token = ""
token_path = Path("hf_temp_token.txt")
if token_path.exists():
    hf_token = token_path.read_text(encoding="utf-8").strip()

smoke_script = f"""#!/usr/bin/env bash
echo "======================================================================"
echo "STARTING 20-EPISODE REMEDIATION SMOKE TEST (ARM O + REMEDIATION)"
echo "Tasks: Dual-Hypothesis Gating, Sensor Fallback, Tool Sanitizer, Loop Warning"
echo "Host: $(hostname), Date: $(date)"
echo "Active GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "======================================================================"

export PYTHONPATH="/home/jovyan/scratch/UAV:$PYTHONPATH"
cd /home/jovyan/scratch/UAV

echo ""
echo ">>> [STEP 1/5] APPLYING REMEDIATION PATCH"
git checkout -f
git clean -fd

cat << 'EOF' | base64 -d | zcat | git apply -
{patch_b64}
EOF

/home/jovyan/.conda/envs/habitat/bin/python -c "
import json
from pathlib import Path
src = Path('agenticnav_paper_replica/data/r2rce_val_unseen_100_rand.json')
dst = Path('agenticnav_paper_replica/data/r2rce_val_unseen_20_remediation.json')
tids = [
    'r2r_ce_val_unseen_0010', 'r2r_ce_val_unseen_0012', 'r2r_ce_val_unseen_0023',
    'r2r_ce_val_unseen_0036', 'r2r_ce_val_unseen_0007', 'r2r_ce_val_unseen_0026',
    'r2r_ce_val_unseen_0039', 'r2r_ce_val_unseen_0049', 'r2r_ce_val_unseen_0054',
    'r2r_ce_val_unseen_0071', 'r2r_ce_val_unseen_0017', 'r2r_ce_val_unseen_0005',
    'r2r_ce_val_unseen_0016', 'r2r_ce_val_unseen_0040', 'r2r_ce_val_unseen_0042',
    'r2r_ce_val_unseen_0013', 'r2r_ce_val_unseen_0058', 'r2r_ce_val_unseen_0031',
    'r2r_ce_val_unseen_0038', 'r2r_ce_val_unseen_0018'
]
with open(src) as f:
    d = json.load(f)
eps_by_id = {{e['episode_id']: e for e in d.get('episodes', [])}}
sel = [eps_by_id[i] for i in tids if i in eps_by_id]
out_d = {{'name': 'r2rce_val_unseen_20_remediation', 'num_episodes': len(sel), 'episodes': sel}}
with open(dst, 'w') as f:
    json.dump(out_d, f, indent=2)
print(f'Generated {{dst}} with {{len(sel)}} episodes.')
"

git add -A
git commit -m "feat(remediation): apply tasks 1-4 remediation for 20ep validation" || true
git status
git log -2 --oneline

echo ""
echo ">>> [STEP 2/5] RUNNING TEST SUITE ON CLUSTER"
/home/jovyan/.conda/envs/habitat/bin/pytest agenticnav_paper_replica/harness/tests/test_remediation_capabilities.py -v
/home/jovyan/.conda/envs/habitat/bin/pytest agenticnav_paper_replica/harness/tests/test_end_to_end_runner.py -v
/home/jovyan/.conda/envs/habitat/bin/pytest agenticnav_paper_replica/geometry/tests/test_safety_and_geometry.py -v

echo ""
echo ">>> [STEP 3/5] EXECUTING 20-EPISODE VALIDATION BENCHMARK"
echo "Dataset: agenticnav_paper_replica/data/r2rce_val_unseen_20_remediation.json (20 episodes)"
echo "Model: qwen3.8:27b"
echo "Output: results/arm_oracle_remediation_20ep"
echo ""

/home/jovyan/.conda/envs/habitat/bin/python scripts/benchmark_r2rce_postfix.py \\
    --model "qwen3.8:27b" \\
    --episodes agenticnav_paper_replica/data/r2rce_val_unseen_20_remediation.json \\
    --output-dir results/arm_oracle_remediation_20ep \\
    --arm O \\
    --locked

echo ""
echo "======================================================================"
echo ">>> [STEP 4/5] 20-EPISODE VALIDATION RESULTS & FAILURE TAXONOMY"
echo "======================================================================"

if [ -f "results/arm_oracle_remediation_20ep/benchmark_summary.json" ]; then
    cat results/arm_oracle_remediation_20ep/benchmark_summary.json | head -n 45
fi

/home/jovyan/.conda/envs/habitat/bin/python -c "
import json
from pathlib import Path

sum_path = Path('results/arm_oracle_remediation_20ep/benchmark_summary.json')
if not sum_path.exists():
    print('Summary not found!')
    exit(0)

with open(sum_path) as f:
    s = json.load(f)

pm = s.get('primary_metrics', {})
sr = pm.get('success_rate', {}).get('mean', 0.0)
osr = pm.get('oracle_success_rate', {}).get('mean', 0.0)
spl = pm.get('spl', {}).get('mean', 0.0)
ne = pm.get('navigation_error_m', {}).get('mean', 0.0)
tax = s.get('failure_taxonomy', {})

print('=== 20-EPISODE REMEDIATION SMOKE TEST RESULTS ===')
print(f'Episodes Evaluated: {{s.get(\"episode_count\")}}')
print(f'Success Rate (SR): {{sr:.1f}}%')
print(f'Oracle Success Rate (OSR): {{osr:.1f}}%')
print(f'SPL: {{spl:.2f}}')
print(f'Navigation Error (NE): {{ne:.2f}} m')
print(f'Failure Taxonomy:')
for k, v in tax.items():
    print(f'  - {{k}}: {{v}}')

deadlocks = tax.get('safety_deadlock', 0)
violations = tax.get('protocol_violation', 0)
print(f'Safety Deadlocks: {{deadlocks}} (Gate: 0)')
print(f'Protocol Violations: {{violations}} (Gate: 0)')
"

echo ""
echo "======================================================================"
echo ">>> [STEP 5/5] UPLOADING SMOKE TEST TRACES TO HUGGING FACE"
echo "======================================================================"

/home/jovyan/.conda/envs/habitat/bin/python -c "
import os
from pathlib import Path
from huggingface_hub import HfApi

token = '{hf_token}'
repo_id = 'Krut27/agenticnav-benchmark-results'

if token and Path('results/arm_oracle_remediation_20ep').exists():
    print('Uploading 20-episode results to Hugging Face...')
    api = HfApi(token=token)
    api.upload_folder(
        folder_path='results/arm_oracle_remediation_20ep',
        path_in_repo='benchmark_results/arm_oracle_remediation_20ep',
        repo_id=repo_id,
        repo_type='dataset',
    )
    print('Upload complete!')
else:
    print('Token missing or results dir not found, skipping upload.')
"

echo "ALL STEPS COMPLETED!"
"""

def main():
    print(f"Connecting to GPU cluster to submit job (script length: {len(smoke_script)} chars)...")
    try:
        job = cluster_client.submit_job(
            name="smoke-20ep-remediation-arm-o",
            script=smoke_script,
            time_limit_min=60,
        )
        print(f"Job successfully submitted! Job ID: #{job}")
    except Exception as e:
        print(f"Error submitting job: {e}")
        return

    print("Monitoring job progress...")
    last_log_len = 0
    while True:
        try:
            status_info = cluster_client.api_request(f"/jobs/{job}")
            state = status_info.get("status", "unknown") if status_info else "unknown"
            log = cluster_client.get_job_log(job)
            if log and len(log) > last_log_len:
                sys.stdout.write(log[last_log_len:])
                sys.stdout.flush()
                last_log_len = len(log)

            if state in ("succeeded", "failed", "cancelled"):
                print(f"\nJob finished with status: {state}")
                break
        except Exception as e:
            print(f"[Polling notice: {e}]")
        time.sleep(10)

if __name__ == "__main__":
    main()
