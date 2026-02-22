#!/bin/bash
# SAC training on BaseMotion3D environment with dense reward
# Usage: ./run_sac_basemotion3d_dense.sh [seed] [reward_scale]

# SEED=${1:-0}              # Default seed: 0
# REWARD_SCALE=${2:-0.1}    # Default dense reward scale: 0.1

# cd "$(dirname "$0")/.."

# # Activate the monorepo virtual environment
# source "$(dirname "$0")/../../.venv/bin/activate"

for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=sac_basemotion3d \
    env_id="kinder/BaseMotion3D-v0" \
    max_episode_steps=100 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.hidden_size=128
done
