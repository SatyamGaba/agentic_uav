"""
Automatic Remote GPU Session Renewal & Benchmark Continuation.
Polls institutional session dashboard, starts GPU container upon quota reset (15:55),
and resumes 40-episode live VLM evaluation.
"""
import ssl
import time
import json
import urllib.request
import exec_cmd

TOKEN = 'b4bbbe0a02884d46af65379e7b74f337'
DASHBOARD_URL = 'https://172.24.16.175/services/gpu-dashboard/api/session/start'
STATUS_URL = 'https://172.24.16.175/user/krutik/api'

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def start_session():
    req = urllib.request.Request(DASHBOARD_URL, headers={'Authorization': f'token {TOKEN}'}, method='POST')
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10.0) as resp:
            print(f"[RENEWAL] Session start returned HTTP {resp.status}")
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')
        print(f"[RENEWAL] HTTP {e.code}: {body}")
        return False
    except Exception as e:
        print(f"[RENEWAL] Error: {e}")
        return False

def wait_for_server():
    req = urllib.request.Request(STATUS_URL, headers={'Authorization': f'token {TOKEN}'})
    while True:
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=10.0) as resp:
                if resp.status == 200:
                    print("[RENEWAL] User notebook server is running and ready!")
                    return True
        except Exception:
            pass
        print("[RENEWAL] Waiting for user server container to initialize...")
        time.sleep(5.0)

def main():
    print("[RENEWAL] Monitoring cluster quota reset window (15:55)...")
    while True:
        if start_session():
            break
        time.sleep(30.0)

    wait_for_server()

    # Re-launch remaining benchmark
    print("[RUNNER] Resuming 40-episode benchmark on remote NVIDIA L40S...")
    cmd = """cd /home/jovyan/scratch && pkill -f run_framework_benchmark.py || true; sleep 1; nohup python run_framework_benchmark.py --adapter calibrated --backend ollama --model qwen3-vl:8b --split benchmark/seen.json --num-episodes 40 --output benchmark_40ep_qwen3vl_seen.json >> benchmark_40ep_live.log 2>&1 & echo $!"""
    out = exec_cmd.run_remote(cmd, wait_sec=8.0)
    print("[RUNNER] Launched with PID:", out.strip())

if __name__ == '__main__':
    main()
