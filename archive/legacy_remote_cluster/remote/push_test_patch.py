import base64
from pathlib import Path
import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request
from manage_job import fetch_log
import time

local_file = Path(r"C:\Users\Kruti\Desktop\LLM\Team7_Task4_L40S_package\Team7_Task4_L40S_package\tool_alloc\tests\test_l40s.py")
content = local_file.read_bytes()
b64_content = base64.b64encode(content).decode("ascii")

script = f"""#!/bin/bash
set -e
cat << 'EOF' | base64 -d > /home/jovyan/scratch/TEST/tool_alloc/tests/test_l40s.py
{b64_content}
EOF
echo "Updated test_l40s.py successfully. File size: $(wc -c < /home/jovyan/scratch/TEST/tool_alloc/tests/test_l40s.py)"

# Update CHECKSUMS.sha256 on remote
cd /home/jovyan/scratch/TEST
SHA=$(sha256sum tool_alloc/tests/test_l40s.py | awk '{{print $1}}')
sed -i "/tool_alloc\/tests\/test_l40s\.py/d" CHECKSUMS.sha256
echo "$SHA  tool_alloc/tests/test_l40s.py" >> CHECKSUMS.sha256
echo "Updated CHECKSUMS.sha256"
"""

payload = {"name": "patch-test-l40s", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break
