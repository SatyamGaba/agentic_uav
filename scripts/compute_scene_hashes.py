import os
import hashlib
import json

base_dir = "/home/krutik/data/scene_datasets/mp3d"
val_scenes = [
    "2azQ1b91cZZ", "8194nk5LbLH", "EU6Fwq7SyZv", "QUCTc6BB5sX", "TbHJrupSAjP",
    "X7HyMhZNoso", "Z6MFQCViBuw", "oLBMNvg9in8", "pLe4wQe7qrG", "x8F5xyUWy9e", "zsNo4HB9uLZ"
]

def get_file_info(path):
    if not os.path.exists(path):
        return None
    size = os.path.getsize(path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            sha256.update(chunk)
    return {
        "size_bytes": size,
        "sha256": sha256.hexdigest()
    }

results = {}
for s in sorted(val_scenes):
    glb_path = os.path.join(base_dir, s, f"{s}.glb")
    nav_path = os.path.join(base_dir, s, f"{s}.navmesh")
    results[s] = {
        "glb": get_file_info(glb_path),
        "navmesh": get_file_info(nav_path)
    }
    print(f"Scene: {s}")
    if results[s]["glb"]:
        print(f"  GLB:     {results[s]['glb']['size_bytes']:>10} bytes | {results[s]['glb']['sha256']}")
    else:
        print("  GLB:     MISSING")
    if results[s]["navmesh"]:
        print(f"  NAVMESH: {results[s]['navmesh']['size_bytes']:>10} bytes | {results[s]['navmesh']['sha256']}")
    else:
        print("  NAVMESH: MISSING")

out_file = "audit/interactive_session/mp3d_val_unseen_11_scenes_hashes.json"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved hash manifest to {out_file}")

