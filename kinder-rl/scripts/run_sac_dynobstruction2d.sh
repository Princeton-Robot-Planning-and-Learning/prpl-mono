#!/bin/bash
# SAC training on DynObstruction2D environment
# Usage: ./run_sac_dynobstruction2d.sh [num_obstructions] [seed]

# NUM_OBS=${1:-1}  # Default: 1 obstruction
# SEED=${2:-0}     # Default seed: 0

# cd "$(dirname "$0")/.."

# # Activate the monorepo virtual environment
# source "$(dirname "$0")/../../.venv/bin/activate"

for seed in 302 301 299
do
python experiments/run_experiment.py \
    agent=sac_dynobstruction2d \
    env_id="kinder/DynObstruction2D-o1-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.hidden_size=128
done