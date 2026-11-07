import sys
sys.path.append(r"C:\Users\Kruti\Desktop\PROG\UAV\remote")
from config import DASHBOARD_API, api_request
from manage_job import fetch_log
import time

script = """#!/bin/bash
cd /home/jovyan/scratch/TEST/tool_alloc
python3 -c "
import json, subprocess, sys, tempfile, yaml
from pathlib import Path
ROOT = Path('/home/jovyan/scratch/TEST/tool_alloc').resolve()
sys.path.insert(0, '/home/jovyan/scratch/TEST')
sys.path.insert(0, str(ROOT))
from core.mock import MockRunner
from core.tree import TreeBuilder
from core.instances import generate_instances

with tempfile.TemporaryDirectory() as td:
    tmp_path = Path(td)
    ins = generate_instances(240, seed=21, prefix='mock')
    tree = TreeBuilder(MockRunner(), [1, 2], log=lambda *_: None).build(ins, tmp_path, 'mock')
    tree.save(tmp_path / 'mock_tree.npz')
    cfg = {'name': 'mock', 'layers': [1, 2], 'lambdas': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
           'splits': {'train': 120, 'cal': 40, 'gate': 30, 'test': 50}, 'partition_seeds': [1, 2],
           'gate': {'threshold_points': 2.0}, 'reference_lambda': 0.3, 'bootstrap': {'n_resamples': 300, 'seed': 0, 'ci_level': 0.95},
           'model': {'id': 'mock', 'revision': 'mock-revision', 'dtype': 'float32'}, 'pool': {'n': 240, 'seed': 21}}
    (tmp_path / 'mock.yaml').write_text(yaml.safe_dump(cfg))
    out = '_pytest_regression'
    r = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'run_experiments.py'), '--config', str(tmp_path / 'mock.yaml'), '--cache-dir', str(tmp_path),
                        '--out', out], capture_output=True, text=True, cwd=ROOT, timeout=900)
    got = json.loads((ROOT / 'results' / out / 'results.json').read_text())
    import shutil
    shutil.rmtree(ROOT / 'results' / out, ignore_errors=True)
    ref = json.loads((ROOT / 'tests' / 'fixtures' / 'ref_mock_results_pre_l40s.json').read_text())
    for d in (got,):
        d.pop('manifest', None)
        for k in ('label_generation_seconds', 'parser', 'parse_categories', 'resolved_revision', 'tree_env', 'nodes_reused_from_cache', 'nodes_generated'):
            d['pool_stats'].pop(k, None)
        for pk in ('partition_0', 'partition_1'):
            d[pk].pop('secondary_contrasts_test_holm', None)
            d[pk].pop('extra_exploratory_contrasts_test', None)

    def diff(a, b, path=''):
        if type(a) != type(b):
            print(f'Type mismatch at {path}: {type(a)} vs {type(b)}')
            return
        if isinstance(a, dict):
            for k in set(a.keys()) ^ set(b.keys()):
                print(f'Key diff at {path}: {k}')
            for k in set(a.keys()) & set(b.keys()):
                diff(a[k], b[k], f'{path}.{k}')
        elif isinstance(a, list):
            if len(a) != len(b):
                print(f'Len diff at {path}: {len(a)} vs {len(b)}')
            for i, (x, y) in enumerate(zip(a, b)):
                diff(x, y, f'{path}[{i}]')
        elif isinstance(a, float):
            if abs(a - b) > 1e-4:
                print(f'Float diff at {path}: got {a} vs ref {b} (diff {abs(a-b)})')
            elif a != b:
                print(f'Micro float diff at {path}: got {a} vs ref {b} (diff {abs(a-b)})')
        elif a != b:
            print(f'Value diff at {path}: got {a} vs ref {b}')

    diff(got, ref)
    print('Done comparing')
"
"""

payload = {"name": "debug-diff-3", "gpu_count": 1, "time_limit_min": 5, "script": script}
res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
jid = res.get("id") or res.get("job_id")
print(f"Submitted Job #{jid}")
while True:
    time.sleep(2)
    j = api_request(f"{DASHBOARD_API}/jobs/{jid}")
    if j.get("status") in ["succeeded", "failed"]:
        print(fetch_log(jid))
        break
