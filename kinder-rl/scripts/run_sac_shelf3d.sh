#!/bin/bash
# SAC training on Shelf3D environment
# Usage: ./run_sac_shelf3d.sh [num_cubes] [seed]

# NUM_CUBES=${1:-1}  # Default: 1 cube
# SEED=${2:-0}       # Default seed: 0

# cd "$(dirname "$0")/.."

# # Activate the monorepo virtual environment
# source "$(dirname "$0")/../../.venv/bin/activate"

for seed in 300 301 302 303 304
do
python experiments/run_experiment.py \
    agent=sac_shelf3d \
    env_id="kinder/TidyBot3D-cupboard_real-o1-v0" \
    max_episode_steps=500 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.hidden_size=256 \
    agent.args.async_envs=true
done