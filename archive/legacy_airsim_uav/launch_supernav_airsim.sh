#!/bin/bash
cd /home/jovyan/scratch
PYTHON=/home/jovyan/scratch/conda_envs/openfly/bin/python

echo "Starting SuperNav AirSim Benchmark (20 Episodes)..."
nohup $PYTHON -u run_framework_benchmark.py \
  --adapter supernav \
  --backend ollama \
  --model qwen3-vl:8b \
  --env airsim \
  --split none \
  --num-episodes 20 \
  --max-steps 10 \
  --output supernav_airsim_20ep_v2.json </dev/null > supernav_airsim_20ep_v2.log 2>&1 &
disown

echo "SuperNav 20-Ep Benchmark launched completely detached. PID: $!"
