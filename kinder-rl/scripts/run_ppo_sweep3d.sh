#!/bin/bash
for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=ppo_tidybot3d \
    env_id="kinder/SweepIntoDrawer3D-o5-v0" \
    max_episode_steps=500 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=64 \
    agent.args.hidden_size=256 \
    agent.args.async_envs=true
done