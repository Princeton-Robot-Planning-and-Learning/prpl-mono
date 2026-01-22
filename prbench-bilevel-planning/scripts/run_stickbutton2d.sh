#!/bin/bash
# Script to run bilevel planning on StickButton2D with multiple random seeds

# Navigate to the prbench-bilevel-planning directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(0 1 2 3 4)

echo "Starting bilevel planning StickButton2D experiments..."
echo "=============================================="

# Run experiments for 1 button
echo ""
echo "Running experiments for 1 button (b1)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        env=stickbutton2d-b1 \
        seed=${seed} \
        hydra.run.dir=./logs/stickbutton2d-b1/seed_${seed}

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
        env=stickbutton2d-b3 \
        seed=${seed} \
        hydra.run.dir=./logs/stickbutton2d-b3/seed_${seed}

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed!"
