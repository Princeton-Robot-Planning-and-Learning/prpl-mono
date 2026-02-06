#!/bin/bash
# SAC training on DynPushPullHook2D environment
# Usage: ./run_sac_dynpushpullhook2d.sh [num_cubes] [seed]

NUM_CUBES=${1:-1}  # Default: 1 cube
SEED=${2:-0}       # Default seed: 0

cd "$(dirname "$0")/.."

# Activate the monorepo virtual environment
source "$(dirname "$0")/../../.venv/bin/activate"

python experiments/run_experiment.py \
    agent=sac_dynpushpullhook2d \
    env_id="kinder/DynPushPullHook2D-o${NUM_CUBES}-v0" \
    max_episode_steps=300 \
    eval_episodes=50 \
    seed=${SEED} \
    agent.args.total_timesteps=1000000 \
    agent.args.hidden_size=128
