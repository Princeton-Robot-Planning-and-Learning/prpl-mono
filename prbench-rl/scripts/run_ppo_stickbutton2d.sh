#!/bin/bash
# Script to run PPO training on stickbutton2d for 0 and 2 buttons with multiple random seeds

# Navigate to the prbench-rl directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(0 1 2 3 4)

echo "Starting PPO stickbutton2d training experiments..."
echo "=============================================="

# Run experiments for 1 buttons
echo ""
echo "Running experiments for 1 buttons (b1)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        agent=ppo_stickbutton2d_1_button \
        env_id=prbench/StickButton2D-b1-v0 \
        seed=${seed} \
        agent.exp_name="ppo_s2d_1_button_seed_${seed}"

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

# Run experiments for 3 buttons
echo ""
echo "Running experiments for 3 buttons (b3)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        agent=ppo_stickbutton2d_3_buttons \
        env_id=prbench/StickButton2D-b3-v0 \
        seed=${seed} \
        agent.exp_name="ppo_s2d_3_button_seed_${seed}"

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed!"
