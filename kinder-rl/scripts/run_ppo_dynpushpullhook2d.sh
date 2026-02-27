# Activate the monorepo virtual environment
source "$(dirname "$0")/../../.venv/bin/activate"

for seed in 301 302 303 304 305
do
python experiments/run_experiment.py \
    agent=ppo_dynpushpullhook2d \
    env_id="kinder/DynPushPullHook2D-o5-v0" \
    max_episode_steps=500 \
    eval_episodes=50 \
    seed=${seed} \
    agent.args.total_timesteps=1000000 \
    agent.args.num_envs=16 \
    agent.args.num_steps=256 \
    agent.args.hidden_size=128
done