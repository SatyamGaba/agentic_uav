import urllib.request
import json

datasets = ['honghd16/mp3d', 'honglyhly/SimpleVLN_SCENCE_DATA', 'bingCS/mp3d_dataset']
for ds in datasets:
    try:
        url = f'https://huggingface.co/api/datasets/{ds}'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=8) as r:
            data = json.loads(r.read().decode())
            private = data.get('private', False)
            siblings = data.get('siblings', [])
            print(f'[FOUND] {ds}: private={private}, files_count={len(siblings)}')
            files = [s['rfilename'] for s in siblings if any(ext in s['rfilename'] for ext in ['.glb', '.navmesh', 'gZ6', 'pLe', 'r1Q', 'zip', 'tar'])]
            print(f'   Target files: {files[:10]}')
    except Exception as e:
        print(f'[ERR] {ds}: {e}')

