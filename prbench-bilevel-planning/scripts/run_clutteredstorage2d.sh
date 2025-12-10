#!/bin/bash
# Script to run bilevel planning on ClutteredStorage2D with multiple random seeds

# Navigate to the prbench-bilevel-planning directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(0 1 2 3 4)

echo "Starting bilevel planning ClutteredStorage2D experiments..."
echo "=============================================="

# Run experiments for 1 block
echo ""
echo "Running experiments for 1 block (b1)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        env=clutteredstorage2d-b1 \
        seed=${seed} \
        hydra.run.dir=./logs/clutteredstorage2d-b1/seed_${seed}

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

# Run experiments for 3 blocks
echo ""
echo "Running experiments for 3 blocks (b3)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        env=clutteredstorage2d-b3 \
        seed=${seed} \
        hydra.run.dir=./logs/clutteredstorage2d-b3/seed_${seed}

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed!"
