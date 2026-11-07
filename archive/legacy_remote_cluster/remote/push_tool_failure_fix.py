import base64
from pathlib import Path
import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request
from manage_job import fetch_log
import time

b_pilot = base64.b64encode(Path(r"C:\Users\Kruti\Desktop\LLM\Team7_Task4_L40S_package\Team7_Task4_L40S_package\tool_alloc\scripts\run_pilot.py").read_bytes()).decode("ascii")
b_exp = base64.b64encode(Path(r"C:\Users\Kruti\Desktop\LLM\Team7_Task4_L40S_package\Team7_Task4_L40S_package\tool_alloc\scripts\run_experiments.py").read_bytes()).decode("ascii")

script = f"""#!/bin/bash
set -e
cat << 'EOF' | base64 -d > /home/jovyan/scratch/TEST/tool_alloc/scripts/run_pilot.py
{b_pilot}
EOF

cat << 'EOF' | base64 -d > /home/jovyan/scratch/TEST/tool_alloc/scripts/run_experiments.py
{b_exp}
EOF

cd /home/jovyan/scratch/TEST
SHA_P=$(sha256sum tool_alloc/scripts/run_pilot.py | awk '{{print $1}}')
SHA_E=$(sha256sum tool_alloc/scripts/run_experiments.py | awk '{{print $1}}')

sed -i "/tool_alloc\/scripts\/run_pilot\.py/d" CHECKSUMS.sha256
sed -i "/tool_alloc\/scripts\/run_experiments\.py/d" CHECKSUMS.sha256

echo "$SHA_P  tool_alloc/scripts/run_pilot.py" >> CHECKSUMS.sha256
echo "$SHA_E  tool_alloc/scripts/run_experiments.py" >> CHECKSUMS.sha256
echo "Updated CHECKSUMS.sha256 successfully"
"""

payload = {"name": "patch-tool-failure", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break

