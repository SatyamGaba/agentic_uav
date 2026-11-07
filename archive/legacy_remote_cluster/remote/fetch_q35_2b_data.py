import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from manage_job import fetch_log
from config import DASHBOARD_API, api_request
import time, base64

script = """#!/bin/bash
echo '=== Q35_RES_START ==='
base64 -w 0 /home/jovyan/scratch/TEST/tool_alloc/results/q35_2b/results.json
echo ''
echo '=== Q35_RES_END ==='
echo '=== Q35_P2_START ==='
base64 -w 0 /home/jovyan/scratch/TEST/tool_alloc/results/q35_2b/p2_results.json
echo ''
echo '=== Q35_P2_END ==='
"""
payload = {'name': 'fetch-q35-2b-results', 'gpu_count': 1, 'time_limit_min': 5, 'script': script}
res = api_request(f'{DASHBOARD_API}/jobs', method='POST', payload=payload)
jid = res.get('id') or res.get('job_id')
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f'{DASHBOARD_API}/jobs/{jid}')
    if j.get('status') in ['succeeded', 'failed']:
        log = fetch_log(jid)
        if '=== Q35_RES_START ===' in log:
            b64_str = log.split('=== Q35_RES_START ===')[1].split('=== Q35_RES_END ===')[0].strip()
            raw = base64.b64decode(b64_str)
            with open('C:/Users/Kruti/Desktop/LLM/results_q35_2b.json', 'wb') as f:
                f.write(raw)
            print('Successfully saved C:/Users/Kruti/Desktop/LLM/results_q35_2b.json! Bytes:', len(raw))
        if '=== Q35_P2_START ===' in log:
            b64_p2 = log.split('=== Q35_P2_START ===')[1].split('=== Q35_P2_END ===')[0].strip()
            raw_p2 = base64.b64decode(b64_p2)
            with open('C:/Users/Kruti/Desktop/LLM/p2_q35_2b.json', 'wb') as f:
                f.write(raw_p2)
            print('Successfully saved C:/Users/Kruti/Desktop/LLM/p2_q35_2b.json! Bytes:', len(raw_p2))
        break

