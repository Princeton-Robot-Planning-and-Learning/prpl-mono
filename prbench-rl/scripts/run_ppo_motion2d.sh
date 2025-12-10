#!/bin/bash
# Script to run PPO training on Motion2D for 0 and 2 passages with multiple random seeds

# Navigate to the prbench-rl directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(0 1 2 3 4)

echo "Starting PPO Motion2D training experiments..."
echo "=============================================="

# Run experiments for 0 passages
echo ""
echo "Running experiments for 0 passages (p0)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        agent=ppo_motion2d_0_passage \
        env_id=prbench/Motion2D-p0-v0 \
        seed=${seed} \
        agent.exp_name="ppo_m2d_0_passage_seed_${seed}"

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

# Run experiments for 2 passages
echo ""
echo "Running experiments for 2 passages (p2)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        agent=ppo_motion2d_2_passages \
        env_id=prbench/Motion2D-p2-v0 \
        seed=${seed} \
        agent.exp_name="ppo_m2d_2_passage_seed_${seed}"

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed!"
