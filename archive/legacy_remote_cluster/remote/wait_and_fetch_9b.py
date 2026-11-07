import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time, base64

job_id = 380
print(f"Monitoring Job #{job_id} until completion...", flush=True)

while True:
    try:
        j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
        status = j.get("status")
        elapsed = j.get("elapsed_s", 0)
        cpu = j.get("cpu_percent")
        idle_s = j.get("gpu_idle_s", 0)
        print(f"Job #{job_id}: {status} (elapsed: {elapsed:.1f}s / {elapsed/60:.1f} min, cpu: {cpu}%, gpu_idle_s: {idle_s:.1f}s)", flush=True)
        
        if status in ["succeeded", "failed", "stopped"]:
            print(f"\nJob #{job_id} reached terminal state: {status} (exit: {j.get('exit_code')}, end_reason: {j.get('end_reason')})", flush=True)
            print("--- LOG ---", flush=True)
            print(fetch_log(job_id), flush=True)
            break
    except Exception as e:
        print(f"Polling warning: {e}", flush=True)
    time.sleep(30)

if status == "succeeded":
    print("\nRetrieving results for q35_9b...", flush=True)
    fetch_script = """#!/bin/bash
echo '=== Q35_9B_RES_START ==='
base64 -w 0 /home/jovyan/scratch/TEST/tool_alloc/results/q35_9b/results.json || true
echo ''
echo '=== Q35_9B_RES_END ==='
echo '=== Q35_9B_P2_START ==='
base64 -w 0 /home/jovyan/scratch/TEST/tool_alloc/results/q35_9b/p2_results.json || true
echo ''
echo '=== Q35_9B_P2_END ==='
"""
    payload = {'name': 'fetch-q35-9b-results', 'gpu_count': 1, 'time_limit_min': 5, 'script': fetch_script}
    res = api_request(f'{DASHBOARD_API}/jobs', method='POST', payload=payload)
    jid = res.get('id') or res.get('job_id')
    while True:
        time.sleep(3)
        j2 = api_request(f'{DASHBOARD_API}/jobs/{jid}')
        if j2.get('status') in ['succeeded', 'failed']:
            log = fetch_log(jid)
            if '=== Q35_9B_RES_START ===' in log:
                b64_str = log.split('=== Q35_9B_RES_START ===')[1].split('=== Q35_9B_RES_END ===')[0].strip()
                if b64_str:
                    raw = base64.b64decode(b64_str)
                    with open('C:/Users/Kruti/Desktop/LLM/results_q35_9b.json', 'wb') as f:
                        f.write(raw)
                    print('Successfully saved C:/Users/Kruti/Desktop/LLM/results_q35_9b.json! Bytes:', len(raw), flush=True)
            if '=== Q35_9B_P2_START ===' in log:
                b64_p2 = log.split('=== Q35_9B_P2_START ===')[1].split('=== Q35_9B_P2_END ===')[0].strip()
                if b64_p2:
                    raw_p2 = base64.b64decode(b64_p2)
                    with open('C:/Users/Kruti/Desktop/LLM/p2_q35_9b.json', 'wb') as f:
                        f.write(raw_p2)
                    print('Successfully saved C:/Users/Kruti/Desktop/LLM/p2_q35_9b.json! Bytes:', len(raw_p2), flush=True)
            break
