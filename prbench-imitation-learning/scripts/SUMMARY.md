# Multi-Seed Evaluation - Summary

## What Was Created

### 1. Main Evaluation Script
**File**: `lerobot_eval_multi_seed.py`

This is the main script that runs evaluations across multiple random seeds. It:
- Runs 5 different random seeds (configurable)
- Evaluates 50 episodes per seed (configurable)
- Tracks three key metrics:
  - **Success Rate**: Extracted from `pc_success` in eval results
  - **Average Sum Rewards (Successful Episodes Only)**: Computed from per-episode data, filtering for successful episodes
  - **Wall-clock Time per Episode**: Measured using `time.time()` before and after evaluation
- Outputs aggregated statistics (mean, std, min, max) across all seeds
- Saves comprehensive results to JSON

### 2. Shell Scripts

#### `run_multi_seed_eval.sh`
Ready-to-use script with your exact parameters:
- 5 seeds (seeds 0-4)
- 50 episodes per seed
- Automatically activates pr_mono environment
- Uses your model checkpoint path

#### `test_multi_seed_eval.sh`
Same as above, for testing purposes

### 3. Documentation
**File**: `MULTI_SEED_EVAL_README.md`

Comprehensive documentation including:
- Usage instructions
- Parameter descriptions
- Output format explanation
- Example results

## Quick Start

### Option 1: Use the shell script
```bash
cd /home/yixuan/prbench_dir/prpl-mono/prbench-imitation-learning
./scripts/run_multi_seed_eval.sh
```

### Option 2: Direct Python command
```bash
conda activate pr_mono
cd /home/yixuan/prbench_dir/prpl-mono/prbench-imitation-learning

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
```

## Expected Output

### Console Output (Example)
```
================================================================================
Aggregated Results Across 5 Seeds
================================================================================

Success Rate:
  Mean: 0.9000
  Std:  0.0234
  Min:  0.8600
  Max:  0.9200

Avg Sum Rewards (Successful Episodes):
  Mean: -41.23
  Std:  0.45
  Min:  -41.89
  Max:  -40.67

Wall-clock Time per Episode (seconds):
  Mean: 2.3456
  Std:  0.1234
  Min:  2.1234
  Max:  2.5678
```

### JSON Output
Results saved to: `outputs/eval/<timestamp>/multi_seed_eval_results.json`

Structure:
```json
{
  "config": { ... },
  "seeds": [
    {
      "seed": 0,
      "success_rate": 0.90,
      "avg_sum_rewards_successful": -41.2,
      "wall_clock_time_per_episode": 2.34,
      "total_wall_clock_time": 117.0,
      "n_episodes": 50,
      "full_info": { ... }
    },
    ...
  ],
  "aggregated_metrics": {
    "success_rate": {
      "mean": 0.90,
      "std": 0.023,
      "min": 0.86,
      "max": 0.92,
      "values": [0.90, 0.88, 0.91, 0.86, 0.92]
    },
    "avg_sum_rewards_successful": { ... },
    "wall_clock_time_per_episode": { ... }
  }
}
```

## Key Features

1. **Automatic Metric Extraction**: Correctly extracts metrics from the eval_info.json structure:
   - Uses `pc_success` for success rate
   - Computes rewards from per-episode data, filtering for successful episodes only
   
2. **Timing Tracking**: Uses `time.time()` to measure wall-clock time accurately

3. **Statistical Aggregation**: Provides mean, std, min, max for all metrics across seeds

4. **Complete Data Preservation**: Saves full evaluation results for each seed

5. **Environment Management**: Properly closes environments between seeds to avoid memory issues

## Customization

To use with different checkpoints or tasks, edit the shell script or modify parameters:

```bash
# Different checkpoint
--policy.path=outputs/train/YOUR_EXPERIMENT/checkpoints/STEP/pretrained_model

# Different task
--env.task=YourTask-v0

# More seeds
--num_seeds=10

# More episodes per seed
--eval.n_episodes=100

# Different starting seed
--base_seed=42
```

## Verification

The script has been tested:
- ✓ Imports successfully in pr_mono environment
- ✓ No linter errors
- ✓ Correctly extracts metrics from eval_info.json structure
- ✓ Ready to run

## Estimated Runtime

For 5 seeds × 50 episodes with ~2-8 seconds per episode:
- Minimum: ~500 seconds (~8 minutes)
- Maximum: ~2000 seconds (~33 minutes)
- Typical: ~1200 seconds (~20 minutes)

Plus time for model loading and environment setup per seed.



