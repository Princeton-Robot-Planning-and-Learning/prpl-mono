#!/bin/bash
# PPO training on DynObstruction2D environment
# Usage: ./run_ppo_dynobstruction2d.sh [num_obstructions] [seed]

# NUM_OBS=${1:-1}  # Default: 1 obstruction
# SEED=${2:-0}     # Default seed: 0

# cd "$(dirname "$0")/.."

# # Activate the monorepo virtual environment
# source "$(dirname "$0")/../../.venv/bin/activate"

for seed in 0 1 2 3 4
do
python experiments/run_experiment.py \
    agent=ppo_basemotion3d \
    env_id="prbench/DynObstruction2D-o1-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=256 \
    agent.args.hidden_size=128
done
