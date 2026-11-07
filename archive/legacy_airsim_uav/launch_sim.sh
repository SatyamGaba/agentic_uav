#!/bin/bash
export LD_LIBRARY_PATH=/home/jovyan/scratch/conda_envs/openfly/lib:$LD_LIBRARY_PATH
export PATH=/home/jovyan/scratch/conda_envs/openfly/bin:/home/jovyan/.local/bin:$PATH
export DISPLAY=:99
export XDG_RUNTIME_DIR=/tmp/runtime-jovyan
mkdir -p /tmp/runtime-jovyan
chmod 700 /tmp/runtime-jovyan

echo "[1/3] Stopping previous instances..."
pkill -9 -f Xvfb
pkill -9 -f AirVLN
sleep 1

echo "[2/3] Starting Xvfb virtual display on :99..."
nohup /home/jovyan/scratch/conda_envs/openfly/bin/Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset > /tmp/xvfb.log 2>&1 &
sleep 2

echo "[3/3] Launching AirVLN Simulator..."
SIM_DIR="/home/jovyan/scratch/OpenFly-Platform/envs/airsim/env_airsim_18/LinuxNoEditor/AirVLN/Binaries/Linux"
chmod +x "$SIM_DIR/AirVLN-Linux-Shipping"
cd "$SIM_DIR"
nohup ./AirVLN-Linux-Shipping AirVLN -windowed -nosound -ResX=1280 -ResY=720 > /home/jovyan/scratch/airvln_sim.log 2>&1 &
sleep 5
ps aux | grep -v grep | grep -E "AirVLN|Xvfb"
