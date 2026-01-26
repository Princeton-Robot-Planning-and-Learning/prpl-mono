#!/bin/bash
# PPO training on Transport3D environment
# Usage: ./run_ppo_transport3d.sh [num_cubes] [seed]

NUM_CUBES=${1:-1}  # Default: 1 cube
SEED=${2:-0}       # Default seed: 0

cd "$(dirname "$0")/.."

python experiments/run_experiment.py \
    agent=ppo_transport3d \
    env_id="prbench/Transport3D-o${NUM_CUBES}-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=${SEED} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=8 \
    agent.args.num_steps=256 \
    agent.args.hidden_size=128
