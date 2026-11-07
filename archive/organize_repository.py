"""
Safe repository reorganization script.
Moves legacy AirSim/UAV and historical one-off files into structured archive directories,
leaving the production AgenticNav pipeline clean, professional, and accessible.
"""

import os
import shutil
from pathlib import Path

ROOT = Path("C:/Users/Kruti/Desktop/PROG/UAV")
ARCHIVE = ROOT / "archive"

LEGACY_AIRSIM = ARCHIVE / "legacy_airsim_uav"
LEGACY_REMOTE = ARCHIVE / "legacy_remote_cluster"
LEGACY_LOGS = ARCHIVE / "legacy_logs_and_scratch"

for d in [LEGACY_AIRSIM, LEGACY_REMOTE, LEGACY_LOGS]:
    d.mkdir(parents=True, exist_ok=True)

# 1. Folders to move to legacy_airsim_uav
airsim_dirs = ["controllers", "multiauav_plat", "worlds", "framework", "eval", "benchmark", "test", "tests", "dependencies"]
for d_name in airsim_dirs:
    src = ROOT / d_name
    if src.exists() and src.is_dir():
        dst = LEGACY_AIRSIM / d_name
        if not dst.exists():
            print(f"Moving dir: {d_name} -> {dst.relative_to(ROOT)}")
            shutil.move(str(src), str(dst))

# 2. Files to move to legacy_airsim_uav
airsim_files = [
    "docker-compose.openfly.yml",
    "openfly_calibrated_20ep.json",
    "openfly_smartway_20ep.json",
    "launch_pure_llm_airsim.sh",
    "launch_supernav_airsim.sh",
    "launch_sim.sh",
    "test_airsim_status.py",
    "test_pure_llm_airsim.py",
    "check_vllm_airsim.py",
]
for f_name in airsim_files:
    src = ROOT / f_name
    if src.exists() and src.is_file():
        dst = LEGACY_AIRSIM / f_name
        print(f"Moving file: {f_name} -> {dst.relative_to(ROOT)}")
        shutil.move(str(src), str(dst))

# 3. Remote cluster folders and scripts -> legacy_remote_cluster
remote_dirs = ["remote", "remote_backup"]
for d_name in remote_dirs:
    src = ROOT / d_name
    if src.exists() and src.is_dir():
        dst = LEGACY_REMOTE / d_name
        if not dst.exists():
            print(f"Moving dir: {d_name} -> {dst.relative_to(ROOT)}")
            shutil.move(str(src), str(dst))

remote_files = [
    "check_remote_40ep_progress.py",
    "check_remote_env.py",
    "check_remote_modelfile.py",
    "check_remote_models.py",
    "check_remote_ollama_ver.py",
    "check_remote_template.py",
    "connect_remote.py",
    "launch_remote_40ep.py",
    "read_remote_log.py",
    "sync_framework_to_remote.py",
    "sync_from_remote.py",
    "test_remote_1ep_vlm.py",
    "test_remote_framework.py",
    "test_remote_prompt.py",
    "test_remote_thinking.py",
    "test_remote_tools.py",
    "test_remote_vlm.py",
    "upload_to_remote.py",
    "uav_ssh_proxy.py",
    "auto_resume_40ep.py",
    "exec_cmd.py",
]
for f_name in remote_files:
    src = ROOT / f_name
    if src.exists() and src.is_file():
        dst = LEGACY_REMOTE / f_name
        print(f"Moving file: {f_name} -> {dst.relative_to(ROOT)}")
        shutil.move(str(src), str(dst))

# 4. Scratch dir and logs -> legacy_logs_and_scratch
scratch_dir = ROOT / "scratch"
if scratch_dir.exists() and scratch_dir.is_dir():
    dst = LEGACY_LOGS / "scratch"
    if not dst.exists():
        print(f"Moving dir: scratch -> {dst.relative_to(ROOT)}")
        shutil.move(str(scratch_dir), str(dst))

# Historical logs and loose test scripts
loose_patterns = [
    "job_", "mavic_vlm.log", "chat.html", "dump.html", "benchmark_output.json", "abstract.txt",
    "check_pkgs.py", "poll_server_readiness.py", "run_bg_bench.sh", "launch_20ep_bench.sh",
    "run_framework_benchmark.py", "demo_closed_loop.py", "debug_ep2.py", "debug_vlm_tool.py",
]
for item in os.listdir(ROOT):
    src = ROOT / item
    if not src.is_file():
        continue
    # Keep core files
    if item in ["README.md", "RESEARCH_GUIDELINES.md", ".coverage", "organize_repository.py"]:
        continue
    # Check if matches pattern
    should_archive = any(p in item for p in loose_patterns)
    if should_archive or item.startswith("test_") or item.startswith("scratch_"):
        dst = LEGACY_LOGS / item
        print(f"Moving file: {item} -> {dst.relative_to(ROOT)}")
        shutil.move(str(src), str(dst))

print("\nRepository reorganization complete!")
