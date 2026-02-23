#!/bin/bash
# PPO training on DynObstruction2D environment
# Usage: ./run_ppo_dynobstruction2d.sh [num_obstructions] [seed]

# NUM_OBS=${1:-1}  # Default: 1 obstruction
# SEED=${2:-0}     # Default seed: 0

# cd "$(dirname "$0")/.."

# # Activate the monorepo virtual environment
# source "$(dirname "$0")/../../.venv/bin/activate"

# DynObstruction2D
# for seed in 301 302 303 304 305
# do
# python experiments/run_experiment.py \
#     agent=ppo_basemotion3d \
#     env_id="kinder/DynObstruction2D-o1-v0" \
#     max_episode_steps=200 \
#     eval_episodes=50 \
#     seed=${seed} \
#     agent.args.total_timesteps=1000000 \
#     agent.args.num_envs=16 \
#     agent.args.num_steps=256 \
#     agent.args.hidden_size=128
# done

# for seed in 301 302 303 304 305
# do
# python experiments/run_experiment.py \
#     agent=ppo_dynpushpullhook2d \
#     env_id="kinder/DynPushPullHook2D-o5-v0" \
#     max_episode_steps=500 \
#     eval_episodes=50 \
#     seed=${seed} \
#     agent.args.total_timesteps=1000000 \
#     agent.args.num_envs=16 \
#     agent.args.num_steps=256 \
#     agent.args.hidden_size=128
# done

for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=ppo_motion2d_0_passage \
    env_id="kinder/Motion2D-p0-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=256 \
    agent.args.hidden_size=128
done

for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=ppo_stickbutton2d_1_button \
    env_id="kinder/StickButton2D-b1-v0" \
    max_episode_steps=200 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=256 \
    agent.args.hidden_size=128
done

# for seed in 301 302 303 304 305
# do
# python experiments/run_experiment.py \
#     agent=ppo_transport3d \
#     env_id="kinder/Transport3D-o2-v0" \
#     max_episode_steps=1200 \
#     eval_episodes=50 \
#     seed=${seed} \
#     agent.args.total_timesteps=1000000 \
#     agent.args.num_envs=16 \
#     agent.args.num_steps=256 \
#     agent.args.hidden_size=128
# done

# for seed in 301 302 303 304 305
# do
# python experiments/run_experiment.py \
#     agent=ppo_tidybot3d \
#     env_id="kinder/TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0" \
#     max_episode_steps=500 \
#     eval_episodes=50 \
#     seed=${seed} \
#     agent.args.total_timesteps=1000000 \
#     agent.args.num_envs=16 \
#     agent.args.num_steps=64 \
#     agent.args.hidden_size=256 \
#     agent.args.async_envs=true
# done

# for seed in 301 302 303 304 305
# do
# python experiments/run_experiment.py \
#     agent=ppo_shelf3d \
#     env_id="kinder/TidyBot3D-cupboard_real-o1-v0" \
#     max_episode_steps=500 \
#     eval_episodes=50 \
#     seed=${seed} \
#     agent.args.total_timesteps=1000000 \
#     agent.args.num_envs=16 \
#     agent.args.num_steps=64 \
#     agent.args.hidden_size=256 \
#     agent.args.async_envs=true
# done