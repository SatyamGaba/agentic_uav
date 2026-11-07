#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export GALLIUM_DRIVER=d3d12
export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

echo "================================================================================"
echo "RUNNING REGRESSION SMOKE TEST (G10.1)"
echo "================================================================================"

/home/krutik/miniconda3/envs/habitat/bin/python "${PROJECT_ROOT}/scripts/regression_smoke.py"

echo "Regression smoke test passed successfully."

