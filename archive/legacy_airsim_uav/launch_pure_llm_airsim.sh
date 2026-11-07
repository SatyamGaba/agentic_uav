#!/bin/bash
cd /home/jovyan/scratch
PYTHON=/home/jovyan/scratch/conda_envs/openfly/bin/python

echo "Starting Pure LLM AirSim Benchmark (20 Episodes)..."
nohup $PYTHON -u run_framework_benchmark.py \
  --adapter direct_pixel \
  --backend ollama \
  --model qwen3-vl:8b \
  --env airsim \
  --split none \
  --num-episodes 20 \
  --max-steps 6 \
  --output pure_llm_airsim_20ep.json > pure_llm_airsim_20ep.log 2>&1 &

echo "Benchmark running in background. PID: $!"
