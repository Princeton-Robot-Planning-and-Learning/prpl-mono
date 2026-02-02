#!/bin/bash
# PPO training on TidyBot3D tool_use environment (sweep blocks into drawer)
# Usage: ./run_ppo_shelf3d.sh [seed]

# SEED=${1:-0}       # Default seed: 0

# cd "$(dirname "$0")/.."

# # Activate the monorepo virtual environment
# source "$(dirname "$0")/../../.venv/bin/activate"

for seed in 300 301 302 303 304
do
python experiments/run_experiment.py \
    agent=ppo_shelf3d \
    env_id="prbench/TidyBot3D-cupboard_real-o1-v0" \
    max_episode_steps=500 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=64 \
    agent.args.hidden_size=256 \
    agent.args.async_envs=true
done