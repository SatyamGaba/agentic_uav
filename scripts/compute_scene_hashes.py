import hashlib
import os
import json

scenes = [
    '2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'QUCTc6BB5sX', 'TbHJrupSAjP',
    'X7HyMhZNoso', 'Z6MFQCViBuw', 'oLBMNvg9in8', 'x8F5xyUWy9e', 'zsNo4HB9uLZ'
]

base_dir = '/home/krutik/data/scene_datasets/mp3d'
results = {}

for s in sorted(scenes):
    glb_path = os.path.join(base_dir, s, f"{s}.glb")
    nav_path = os.path.join(base_dir, s, f"{s}.navmesh")
    
    glb_hash = "missing"
    if os.path.exists(glb_path):
        h = hashlib.sha256()
        with open(glb_path, 'rb') as f:
            while chunk := f.read(65536):
                h.update(chunk)
        glb_hash = h.hexdigest()
        
    nav_hash = "missing"
    if os.path.exists(nav_path):
        h = hashlib.sha256()
        with open(nav_path, 'rb') as f:
            while chunk := f.read(65536):
                h.update(chunk)
        nav_hash = h.hexdigest()
        
    results[s] = {"glb_sha256": glb_hash, "navmesh_sha256": nav_hash}
    print(f"{s}: glb={glb_hash[:16]} nav={nav_hash[:16]}")

out_path = "audit/metric_validation/locked_100_scene_hashes.json"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print("Saved to", out_path)
