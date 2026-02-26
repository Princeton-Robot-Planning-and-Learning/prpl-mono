#!/bin/bash
# PPO training on TidyBot3D tool_use environment (sweep blocks into drawer)

for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=ppo_tidybot3d \
    env_id="kinder/TidyBot3D-tool_use-lab2_kitchen-o5-sweep_the_blocks_into_the_top_drawer_of_the_kitchen_island-v0" \
    max_episode_steps=600 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=64 \
    agent.args.hidden_size=256 \
    agent.args.async_envs=true
done