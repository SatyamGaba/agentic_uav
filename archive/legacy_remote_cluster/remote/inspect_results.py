import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

script = """#!/bin/bash
cd /home/jovyan/scratch/TEST/tool_alloc
python3 << 'EOF'
import json
res = json.load(open('results/gpu_0p5b/results.json'))
p0 = res['partition_0']
print('=== SECONDARY CONTRASTS (HOLM) ===')
for k, v in p0.get('secondary_contrasts_test_holm', {}).items():
    diff_val = v.get('diff')
    p_raw = v.get('p')
    p_holm = v.get('p_holm')
    sig = v.get('significant_holm')
    print(f"{k}: diff={diff_val}, p_raw={p_raw}, p_holm={p_holm}, sig={sig}")

print('\n=== UTILITY SUMMARY (TEST SET) ===')
for pol, u in p0.get('u_auc_test', {}).items():
    print(f"  {pol:<24}: {u:.3f}")

print('\n=== PROBE SELECTION ===')
print('Chosen layer:', p0.get('probes', {}).get('layer'))
print('Chosen C:', p0.get('probes', {}).get('C'))
EOF
"""

payload = {"name": "inspect-results-2", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break
