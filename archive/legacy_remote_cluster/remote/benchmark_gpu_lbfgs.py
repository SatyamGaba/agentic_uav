import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request
from manage_job import fetch_log
import time

test_script = """#!/bin/bash
python3 -c "
import torch, numpy as np, time
from scipy.optimize import minimize

X = np.random.randn(3360, 4096).astype(np.float32)
y = (np.random.rand(3360) > 0.5).astype(np.float32)
C = 1.0

def _fg(theta, X, y):
    w, b = theta[:-1], theta[-1]
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
    eps = 1e-12
    loss = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + 0.5 / C * np.dot(w, w)
    err = p - y
    return float(loss), np.append(X.T @ err + w / C, err.sum())

t0 = time.time()
res_cpu = minimize(_fg, np.zeros(X.shape[1] + 1), args=(X, y), jac=True, method='L-BFGS-B', options={'maxiter': 50})
t_cpu = time.time() - t0
print(f'SciPy CPU 50 iters: {t_cpu:.3f}s, loss: {res_cpu.fun:.3f}')

t0 = time.time()
Xt = torch.tensor(X, device='cuda')
yt = torch.tensor(y, device='cuda')
w = torch.zeros(4096, device='cuda', requires_grad=True)
b = torch.zeros(1, device='cuda', requires_grad=True)
opt = torch.optim.LBFGS([w, b], max_iter=50, tolerance_grad=1e-5, line_search_fn='strong_wolfe')
def closure():
    opt.zero_grad()
    p = torch.sigmoid(Xt @ w + b)
    eps = 1e-12
    loss = -torch.sum(yt * torch.log(p + eps) + (1 - yt) * torch.log(1 - p + eps)) + 0.5 / C * torch.sum(w * w)
    loss.backward()
    return loss
opt.step(closure)
torch.cuda.synchronize()
t_gpu = time.time() - t0
print(f'PyTorch GPU 50 iters: {t_gpu:.3f}s, loss: {closure().item():.3f}')
print(f'Speedup: {t_cpu / t_gpu:.1f}x!')
"
"""
payload = {'name': 'test-gpu-lbfgs', 'gpu_count': 1, 'time_limit_min': 5, 'script': test_script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get('id') or res.get('job_id')
print(f"Submitted test job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get('status') in ['succeeded', 'failed']:
        print(fetch_log(jid))
        break

