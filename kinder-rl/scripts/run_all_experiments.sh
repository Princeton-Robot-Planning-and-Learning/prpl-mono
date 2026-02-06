#!/bin/bash
# Run all RL experiments across environments
# Usage: ./run_all_experiments.sh [seed]

SEED=${1:-0}
SCRIPT_DIR="$(dirname "$0")"

echo "=============================================="
echo "Running all RL experiments with seed=${SEED}"
echo "=============================================="

# 2D Environments
echo "[1/10] PPO on DynObstruction2D..."
"${SCRIPT_DIR}/run_ppo_dynobstruction2d.sh" 1 ${SEED}

echo "[2/10] SAC on DynObstruction2D..."
"${SCRIPT_DIR}/run_sac_dynobstruction2d.sh" 1 ${SEED}

echo "[3/10] PPO on DynPushPullHook2D..."
"${SCRIPT_DIR}/run_ppo_dynpushpullhook2d.sh" 1 ${SEED}

echo "[4/10] SAC on DynPushPullHook2D..."
"${SCRIPT_DIR}/run_sac_dynpushpullhook2d.sh" 1 ${SEED}

# 3D Environments
echo "[5/10] PPO on Transport3D..."
"${SCRIPT_DIR}/run_ppo_transport3d.sh" 1 ${SEED}

echo "[6/10] SAC on Transport3D..."
"${SCRIPT_DIR}/run_sac_transport3d.sh" 1 ${SEED}

echo "[7/10] PPO on Shelf3D..."
"${SCRIPT_DIR}/run_ppo_shelf3d.sh" 1 ${SEED}

echo "[8/10] SAC on Shelf3D..."
"${SCRIPT_DIR}/run_sac_shelf3d.sh" 1 ${SEED}

# BaseMotion3D with dense reward
echo "[9/10] PPO on BaseMotion3D (dense reward)..."
"${SCRIPT_DIR}/run_ppo_basemotion3d_dense.sh" ${SEED}

echo "[10/10] SAC on BaseMotion3D (dense reward)..."
"${SCRIPT_DIR}/run_sac_basemotion3d_dense.sh" ${SEED}

echo "=============================================="
echo "All experiments completed!"
echo "=============================================="
