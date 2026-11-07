#!/bin/bash
cd /home/jovyan/scratch
export PYTHONPATH=/home/jovyan/scratch:$PYTHONPATH
nohup /home/jovyan/scratch/conda_envs/openfly/bin/python -u run_framework_benchmark.py --adapter calibrated --env airsim --num-episodes 5 --output /home/jovyan/scratch/calibrated_airsim_results.json > /home/jovyan/scratch/calibrated_airsim_live.log 2>&1 &
echo "BENCHMARK_PID: $!"
sleep 2
ps aux | grep -v grep | grep run_framework_benchmark
