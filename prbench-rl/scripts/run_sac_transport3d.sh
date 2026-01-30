#!/bin/bash
# SAC training on Transport3D environment
# Usage: ./run_sac_transport3d.sh [num_cubes] [seed]

# NUM_CUBES=${1:-1}  # Default: 1 cube
# SEED=${2:-0}       # Default seed: 0

# cd "$(dirname "$0")/.."

# # Activate the monorepo virtual environment
# source "$(dirname "$0")/../../.venv/bin/activate"

for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=sac_transport3d \
    env_id="prbench/Transport3D-o2-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.num_envs=16 \
    agent.args.total_timesteps=1000000 \
    agent.args.hidden_size=128
done