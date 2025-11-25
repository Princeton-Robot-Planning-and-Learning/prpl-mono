#!/bin/bash

# Test script for multi-seed evaluation
# This runs a quick test with fewer episodes to verify the script works

# Activate the pr_mono environment
echo "Activating pr_mono environment..."
eval "$(conda shell.bash hook)"
conda activate pr_mono

# Change to the correct directory
cd /home/yixuan/prbench_dir/prpl-mono/prbench-imitation-learning

# Run the multi-seed evaluation with a small number of episodes for testing
echo "Running multi-seed evaluation test..."
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

echo "Test completed! Check the output directory for results."


