import urllib.request
import ssl
import json
import time

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

TOKEN = "b4bbbe0a02884d46af65379e7b74f337"
BASE_URL = "https://172.24.16.175/services/gpu-dashboard/api"
HEADERS = {
    "Authorization": f"token {TOKEN}",
    "Content-Type": "application/json"
}

def api_request(path, method="GET", data=None, timeout=30):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = urllib.request.Request(url, headers=HEADERS, data=body, method=method)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
            content = resp.read().decode("utf-8")
            if content:
                return json.loads(content)
            return None
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        print(f"API HTTPError {e.code}: {err_body}")
        raise

def submit_job(name, script, time_limit_min=5):
    payload = {
        "name": name,
        "script": script,
        "time_limit_min": max(5, time_limit_min)
    }
    res = api_request("/jobs", method="POST", data=payload)
    print(f"Submitted job {res['id']}: {name}")
    return res["id"]

def wait_for_job(job_id, poll_interval=3, max_wait=300):
    start = time.time()
    while time.time() - start < max_wait:
        res = api_request(f"/jobs/{job_id}")
        status = res.get("status")
        print(f"Job {job_id} status: {status}")
        if status in ("succeeded", "failed", "cancelled"):
            return status, res
        time.sleep(poll_interval)
    raise TimeoutError(f"Job {job_id} timed out after {max_wait}s")

def get_job_log(job_id, timeout=40):
    url = f"{BASE_URL}/jobs/{job_id}/log"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")

def run_job_sync(name, script, time_limit_min=5, max_wait=300):
    jid = submit_job(name, script, time_limit_min)
    status, _ = wait_for_job(jid, max_wait=max_wait)
    log = get_job_log(jid)
    print(f"--- LOG FOR JOB {jid} ({status}) ---")
    print(log)
    print(f"--- END LOG FOR JOB {jid} ---")
    return status, log

if __name__ == "__main__":
    jobs = api_request("/jobs")
    job_list = jobs if isinstance(jobs, list) else jobs.get("jobs", [])
    print(f"Total jobs: {len(job_list)}")
    for j in job_list[:10]:
        print(f"ID: {j.get('id')} | Name: {j.get('name')} | Status: {j.get('status')} | Created: {j.get('created_at')}")

