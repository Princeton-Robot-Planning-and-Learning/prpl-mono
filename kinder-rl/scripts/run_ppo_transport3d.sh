#!/bin/bash
# PPO training on Transport3D environment

for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=ppo_transport3d \
    env_id="kinder/Transport3D-o2-v0" \
    max_episode_steps=1200 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=256 \
    agent.args.hidden_size=128
done
