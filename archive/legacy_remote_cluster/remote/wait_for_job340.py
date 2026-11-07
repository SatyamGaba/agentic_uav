import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time

job_id = 340
print(f"Polling Job #{job_id} until completion...")
while True:
    j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
    status = j.get("status")
    print(f"Job #{job_id} {status} (elapsed: {j.get('elapsed_s'):.1f}s, cpu: {j.get('cpu_percent')}%)")
    if status in ["succeeded", "failed"]:
        print(f"Final status: {status}, exit_code: {j.get('exit_code')}")
        print("--- LOG ---")
        print(fetch_log(job_id))
        break
    time.sleep(5)

