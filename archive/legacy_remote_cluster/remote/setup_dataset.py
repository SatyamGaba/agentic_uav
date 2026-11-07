"""
Matterport3D (MP3D) & R2R-CE Dataset Setup, Symlinking & Verification Utility.
Inspects the remote server for existing 3D scene meshes, automatically creates symlinks
into the project workspace, and supports selective download of the 11 val_unseen scenes.

Usage Examples:
    # Scan remote cluster and auto-symlink detected Matterport3D scenes:
    python remote/setup_dataset.py

    # Explicitly symlink from a known shared cluster directory:
    python remote/setup_dataset.py --link-from /data/scene_datasets/mp3d

    # Download the 11 val_unseen scenes directly to the cluster (requires token):
    python remote/setup_dataset.py --download-token YOUR_MATTERPORT_TOKEN
"""
import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from remote.config import DASHBOARD_API, api_request
from remote.manage_job import fetch_log

# The 11 Matterport3D scenes used by R2R-CE val_unseen 100 episodes
VAL_UNSEEN_SCENES = [
    "gZ6f7yhEvPG", "pLe4wQe7qrG", "r1Q1Z4BcVte", "s8pcmisogdp",
    "2t7WUuJeko7", "17DRP5sb8fy", "pa4DalZuwoK", "VLzqgDo317q",
    "YmJkqBEsHnH", "XcA2TqAcquw", "JeFG25nYack"
]

def generate_setup_script(link_from: str = None, download_token: str = None) -> str:
    link_clause = f'manual_link_dir = "{link_from}"' if link_from else 'manual_link_dir = None'
    token_clause = f'download_token = "{download_token}"' if download_token else 'download_token = None'

    script = f"""#!/usr/bin/env bash
set -e
echo "======================================================================"
echo "          MATTERPORT3D / R2R-CE DATASET SETUP & VERIFICATION          "
echo "======================================================================"

TARGET_DIR="/home/jovyan/scratch/agenticnav_paper_replica/data/scene_datasets/mp3d"
mkdir -p "$TARGET_DIR"

echo "1. Checking episode task file..."
EP_FILE="/home/jovyan/scratch/agenticnav_paper_replica/agenticnav_paper_replica/data/episode_list_100.json"
if [ -f "$EP_FILE" ]; then
    echo "  [FOUND] Episode list file exists: $EP_FILE"
    python3 -c "
import json
with open('$EP_FILE') as f:
    d = json.load(f)
print(f'  [VALID] Contains {{len(d.get(\"episodes\", []))}} episodes ready for evaluation.')
"
else
    echo "  [MISSING] Episode list not found at $EP_FILE. Please run: python remote/sync_all.py"
fi

echo -e "\\n2. Scanning for Matterport3D scene meshes (.glb / .navmesh)..."
python3 -c "
import os
import sys

val_scenes = [
    'gZ6f7yhEvPG', 'pLe4wQe7qrG', 'r1Q1Z4BcVte', 's8pcmisogdp',
    '2t7WUuJeko7', '17DRP5sb8fy', 'pa4DalZuwoK', 'VLzqgDo317q',
    'YmJkqBEsHnH', 'XcA2TqAcquw', 'JeFG25nYack'
]

target_dir = '$TARGET_DIR'
{link_clause}
{token_clause}

search_dirs = [
    manual_link_dir,
    '/data/scene_datasets/mp3d',
    '/data/mp3d',
    '/datasets/mp3d',
    '/datasets/scene_datasets/mp3d',
    '/home/jovyan/scratch/scene_datasets/mp3d',
    '/home/jovyan/scratch/data/scene_datasets/mp3d',
    '/home/jovyan/scratch/mp3d',
    target_dir
]
search_dirs = [d for d in search_dirs if d and os.path.isdir(d)]

found_scenes = {{}}
symlinked_count = 0

for base in search_dirs:
    for s in val_scenes:
        if s in found_scenes:
            continue
        p1 = os.path.join(base, s, f'{{s}}.glb')
        p2 = os.path.join(base, f'{{s}}.glb')
        src_path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)

        if src_path:
            found_scenes[s] = src_path
            dest_scene_dir = os.path.join(target_dir, s)
            os.makedirs(dest_scene_dir, exist_ok=True)
            dest_glb = os.path.join(dest_scene_dir, f'{{s}}.glb')

            # Create symlink if not already inside target_dir
            if not os.path.exists(dest_glb):
                try:
                    os.symlink(src_path, dest_glb)
                    symlinked_count += 1
                except Exception as e:
                    pass

            # Also check for .navmesh
            src_nav = os.path.splitext(src_path)[0] + '.navmesh'
            dest_nav = os.path.join(dest_scene_dir, f'{{s}}.navmesh')
            if os.path.exists(src_nav) and not os.path.exists(dest_nav):
                try:
                    os.symlink(src_nav, dest_nav)
                except Exception:
                    pass

print(f'Matterport3D Scan Results:')
print(f'  Target Directory : {{target_dir}}')
print(f'  Available Scenes : {{len(found_scenes)}} / {{len(val_scenes)}}')
if symlinked_count > 0:
    print(f'  Symlinked Scenes : {{symlinked_count}} scenes successfully mapped into workspace!')

if len(found_scenes) == len(val_scenes):
    print('\\n  [SUCCESS] All 11 val_unseen Matterport3D scenes are ready for Habitat-Sim!')
else:
    missing = [s for s in val_scenes if s not in found_scenes]
    print(f'\\n  [NOTICE] {{len(missing)}} scenes currently not found on cluster: {{missing}}')
    if download_token:
        print('  Proceeding to download missing scenes with provided token...')
    else:
        print('  Options to proceed:')
        print('    1. If testing immediately, use --mock flag (runs complete pipeline with MockGroundEmbodiment).')
        print('    2. If you have a Matterport token, run: python remote/setup_dataset.py --download-token <TOKEN>')
"

# 3. Optional Download step if token provided
if [ -n "{download_token or ''}" ]; then
    echo -e "\\n3. Executing selective download of 11 val_unseen scenes..."
    SCENE_LIST_STR="gZ6f7yhEvPG,pLe4wQe7qrG,r1Q1Z4BcVte,s8pcmisogdp,2t7WUuJeko7,17DRP5sb8fy,pa4DalZuwoK,VLzqgDo317q,YmJkqBEsHnH,XcA2TqAcquw,JeFG25nYack"
    
    # Download official Princeton script if not present
    if [ ! -f "download_mp.py" ]; then
        curl -s -L -o download_mp.py "https://raw.githubusercontent.com/niessner/Matterport/master/download_mp.py" || true
    fi

    if [ -f "download_mp.py" ]; then
        python3 download_mp.py -o "$TARGET_DIR" \\
            --id "{download_token}" \\
            --type habitat \\
            --task_data \\
            --scene_ids "$SCENE_LIST_STR"
        echo "Download finished! Verifying target files:"
        ls -la "$TARGET_DIR"
    else
        echo "[ERROR] Could not fetch download_mp.py from repository."
    fi
fi

echo "======================================================================"
echo "          DATASET SETUP & VERIFICATION COMPLETED                      "
echo "======================================================================"
"""
    return script

def main():
    parser = argparse.ArgumentParser(description="Verify and configure Matterport3D dataset on remote cluster.")
    parser.add_argument("--link-from", type=str, default=None,
                        help="Shared directory path containing mp3d scenes to symlink.")
    parser.add_argument("--download-token", type=str, default=None,
                        help="Matterport3D authorized download token for fetching 11 val_unseen scenes.")
    args = parser.parse_args()

    print("=" * 70)
    print("      CONFIGURING DATASETS ON ANANTA GPU CLUSTER                  ")
    print("=" * 70)

    setup_script = generate_setup_script(
        link_from=args.link_from,
        download_token=args.download_token
    )

    payload = {
        "name": "setup-dataset-job",
        "gpu_count": 0,
        "time_limit_min": 15 if args.download_token else 5,
        "script": setup_script,
    }

    try:
        res = api_request(f"{DASHBOARD_API}/jobs", method="POST", payload=payload)
    except Exception as e:
        print(f"[ERROR] Could not submit dataset setup job: {e}")
        print("\nNote: If you receive HTTP 403 Forbidden:")
        print("  1. Please log in to the campus network portal at: https://campnet.bits-goa.ac.in:8090/")
        print("  2. If connecting from outside campus, ensure FortiClient VPN is active.")
        sys.exit(1)

    job_id = res.get("job_id") or res.get("id")
    print(f"[SUCCESS] Dataset setup job submitted (Job ID: #{job_id}). Processing...")

    for _ in range(30):
        time.sleep(3)
        try:
            j = api_request(f"{DASHBOARD_API}/jobs/{job_id}")
            status = j.get("status")
            if status in ["succeeded", "failed"]:
                log = fetch_log(job_id)
                print(log)
                return
        except Exception as e:
            print(f"Waiting: {e}")

    print(f"Job processing in background. View log with: python remote/manage_job.py --id {job_id}")

if __name__ == "__main__":
    main()
