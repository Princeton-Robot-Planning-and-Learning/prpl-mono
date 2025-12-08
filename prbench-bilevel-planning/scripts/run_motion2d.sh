#!/bin/bash
# Script to run bilevel planning on Motion2D with multiple random seeds

# Navigate to the prbench-bilevel-planning directory
cd "$(dirname "$0")/.." || exit 1

# Define the seeds to run
SEEDS=(0 1 2 3 4)

echo "Starting bilevel planning Motion2D experiments..."
echo "=============================================="

# Run experiments for 1 passage
echo ""
echo "Running experiments for 1 passage (p0)..."
for seed in "${SEEDS[@]}"; do
    echo "  - Running seed ${seed}..."
    python experiments/run_experiment.py \
        env=motion2d-p0 \
        seed=${seed} \
        hydra.run.dir=./logs/motion2d-p0/seed_${seed}

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
        env=motion2d-p2 \
        seed=${seed} \
        hydra.run.dir=./logs/motion2d-p2/seed_${seed}

    if [ $? -eq 0 ]; then
        echo "    ✓ Seed ${seed} completed successfully"
    else
        echo "    ✗ Seed ${seed} failed"
    fi
done

echo ""
echo "=============================================="
echo "All experiments completed!"
