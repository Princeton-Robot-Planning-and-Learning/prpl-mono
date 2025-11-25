# Multi-Seed Evaluation Script

This script (`lerobot_eval_multi_seed.py`) runs policy evaluation across multiple random seeds and aggregates the results.

## Features

- Evaluates policy across multiple random seeds (default: 5)
- Runs specified number of episodes per seed (e.g., 50)
- Tracks and logs:
  - **Success Rate**: Percentage of successful episodes
  - **Average Sum Rewards (Successful Episodes)**: Mean reward for successful episodes only
  - **Wall-clock Time per Episode**: Time taken per episode in seconds
- Aggregates metrics across all seeds (mean, std, min, max)
- Saves detailed results in JSON format

## Usage

### Quick Start

Run the provided shell script:

```bash
./scripts/run_multi_seed_eval.sh
```

### Manual Usage

```bash
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

### Parameters

All standard `lerobot_eval.py` parameters are supported, plus:

- `--num_seeds`: Number of random seeds to evaluate (default: 5)
- `--base_seed`: Starting seed value (default: 0). Seeds used will be [base_seed, base_seed+1, ..., base_seed+num_seeds-1]

## Output

### Console Output

The script prints:
1. Progress for each seed
2. Per-seed results (success rate, avg rewards, timing)
3. Final aggregated statistics across all seeds

Example output:
```
================================================================================
Aggregated Results Across 5 Seeds
================================================================================

Success Rate:
  Mean: 0.8540
  Std:  0.0234
  Min:  0.8200
  Max:  0.8800

Avg Sum Rewards (Successful Episodes):
  Mean: 125.34
  Std:  3.45
  Min:  120.12
  Max:  130.45

Wall-clock Time per Episode (seconds):
  Mean: 2.3456
  Std:  0.1234
  Min:  2.1234
  Max:  2.5678
```

### JSON Output

Results are saved to `outputs/eval/<timestamp>/multi_seed_eval_results.json` with structure:

```json
{
  "config": { ... },
  "seeds": [
    {
      "seed": 0,
      "success_rate": 0.85,
      "avg_sum_rewards_successful": 125.5,
      "wall_clock_time_per_episode": 2.34,
      "total_wall_clock_time": 117.0,
      "n_episodes": 50,
      "full_info": { ... }
    },
    ...
  ],
  "aggregated_metrics": {
    "success_rate": {
      "mean": 0.854,
      "std": 0.023,
      "min": 0.82,
      "max": 0.88,
      "values": [0.85, 0.84, 0.86, 0.82, 0.88]
    },
    ...
  }
}
```

### Video Output

Evaluation videos are saved per seed in:
- `outputs/eval/<timestamp>/videos_seed_0/`
- `outputs/eval/<timestamp>/videos_seed_1/`
- etc.

## Environment Setup

Make sure to activate the `pr_mono` environment before running:

```bash
conda activate pr_mono
```

## Customization

To evaluate different checkpoints or tasks, modify the parameters:

```bash
# Different checkpoint
--policy.path=outputs/train/<your_experiment>/checkpoints/<step>/pretrained_model

# Different task
--env.task=YourTask-v0

# More seeds or episodes
--num_seeds=10 \
--eval.n_episodes=100

# Different starting seed
--base_seed=42
```

## Notes

- The script automatically closes environments between seeds to avoid memory issues
- Videos are limited to 10 episodes per seed to save disk space
- Each seed evaluation is independent and can be run in parallel if needed (requires script modification)
- Total time = (num_seeds × n_episodes × time_per_episode)


