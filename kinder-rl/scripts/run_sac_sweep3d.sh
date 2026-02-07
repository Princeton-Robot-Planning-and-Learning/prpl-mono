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
    agent=sac_tidybot3d \
    env_id="kinder/TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0" \
    max_episode_steps=600 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.hidden_size=256 \
    agent.args.async_envs=true
done