#!/bin/bash

# Multi-seed evaluation script
# Runs evaluation with 5 different random seeds and 50 evaluations per seed
# Logs: success rate, average sum_rewards per successful episode, and wall-clock time per episode

# Activate the pr_mono environment
echo "Activating pr_mono environment..."
eval "$(conda shell.bash hook)"
conda activate pr_mono

# Change to the correct directory
cd /home/yixuan/prbench_dir/prpl-mono/prbench-imitation-learning

# Run the multi-seed evaluation
echo "Running multi-seed evaluation with 5 seeds and 50 episodes per seed..."
python scripts/lerobot_eval_multi_seed.py \
    --policy.path=outputs/train/2025-11-19/18-25-35_prbench_diffusion/checkpoints/030000/pretrained_model \
    --env.type=prbench \
    --env.task=Motion2D-p0-v0 \
    --eval.batch_size=20 \
    --eval.n_episodes=50 \
    --policy.use_amp=false \
    --policy.device=cuda \
    --policy.crop_shape=[64,64] \
    --num_seeds=5 \
    --base_seed=0

echo ""
echo "==========================================="
echo "Multi-seed evaluation completed!"
echo "Results saved to: outputs/eval/<timestamp>/multi_seed_eval_results.json"
echo "==========================================="


