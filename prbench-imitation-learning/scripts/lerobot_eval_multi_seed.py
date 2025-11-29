"""Evaluate a policy on an environment across multiple random seeds.

Usage example:

```
python scripts/lerobot_eval_multi_seed.py \
    --policy.path=path/to/pretrained_model \
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
"""

import json
import logging
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import numpy as np
import torch
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.envs.factory import make_env
from lerobot.envs.utils import close_envs
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging
from termcolor import colored

from prbench_imitation_learning.evaluate import eval_policy_all


@dataclass
class MultiSeedEvalConfig(EvalPipelineConfig):
    """Config for multi-seed evaluation."""

    num_seeds: int = 5  # Number of random seeds to evaluate
    base_seed: int = (
        0  # Base seed (will use base_seed, base_seed+1, ..., base_seed+num_seeds-1)
    )


@parser.wrap()
def eval_multi_seed_main(cfg: MultiSeedEvalConfig):
    """Evaluate a policy across multiple random seeds and aggregate metrics."""
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info(  # pylint: disable=logging-not-lazy
        colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}"
    )

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Store results across all seeds
    all_results = {
        "config": asdict(cfg),
        "seeds": [],
        "aggregated_metrics": {},
    }

    # Track metrics across all seeds
    all_success_rates = []
    all_avg_rewards_successful = []
    all_wall_clock_times = []

    # Iterate over seeds
    for seed_idx in range(cfg.num_seeds):
        current_seed = cfg.base_seed + seed_idx
        logging.info(
            colored(
                f"\n{'='*80}\nEvaluating with seed {current_seed} ({seed_idx+1}/{cfg.num_seeds})\n{'='*80}",  # pylint: disable=line-too-long
                "cyan",
                attrs=["bold"],
            )
        )

        set_seed(current_seed)

        # Make environment
        logging.info("Making environment.")
        envs = make_env(
            cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs
        )

        # Make policy (only once per seed to avoid reloading)
        logging.info("Making policy.")
        policy = make_policy(
            cfg=cfg.policy,
            env_cfg=cfg.env,
        )
        policy.eval()

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=cfg.policy,
            pretrained_path=cfg.policy.pretrained_path,
            preprocessor_overrides={
                "device_processor": {"device": str(policy.config.device)}
            },
        )

        # Run evaluation and track time
        start_time = time.time()

        with (
            torch.no_grad(),
            (
                torch.autocast(device_type=device.type)
                if cfg.policy.use_amp
                else nullcontext()
            ),
        ):
            info = eval_policy_all(
                envs=envs,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                n_episodes=cfg.eval.n_episodes,
                max_episodes_rendered=10,
                videos_dir=output_dir / f"videos_seed_{current_seed}",
                start_seed=current_seed,
                max_parallel_tasks=cfg.env.max_parallel_tasks,
            )

        end_time = time.time()
        total_time = end_time - start_time
        time_per_episode = total_time / cfg.eval.n_episodes

        # Close environments
        close_envs(envs)

        # Extract metrics from info
        # The info structure has per_task with individual episode data
        # and overall aggregated metrics

        # Get overall metrics
        overall_info = info.get("overall", {})

        # Extract success rate from pc_success (percentage, so convert to rate)
        success_rate = overall_info.get("pc_success", 0.0) / 100.0

        # Compute average reward for successful episodes only
        # We need to use per_task data to get individual episode results
        all_sum_rewards = []
        all_successes = []

        for task_data in info.get("per_task", []):
            metrics = task_data.get("metrics", {})
            sum_rewards = metrics.get("sum_rewards", [])
            successes = metrics.get("successes", [])

            all_sum_rewards.extend(sum_rewards)
            all_successes.extend(successes)

        # Compute average reward for successful episodes only
        if all_sum_rewards and all_successes:
            successful_rewards = [
                reward
                for reward, success in zip(all_sum_rewards, all_successes)
                if success
            ]
            avg_reward_successful = (
                np.mean(successful_rewards) if successful_rewards else 0.0
            )
        else:
            # Fallback to overall average if per-task data not available
            logging.warning(
                "Could not extract per-episode data, using overall avg_sum_reward"
            )
            avg_reward_successful = overall_info.get("avg_sum_reward", 0.0)

        # Store results for this seed
        seed_result = {
            "seed": current_seed,
            "success_rate": float(success_rate),
            "avg_sum_rewards_successful": float(avg_reward_successful),
            "wall_clock_time_per_episode": float(time_per_episode),
            "total_wall_clock_time": float(total_time),
            "n_episodes": cfg.eval.n_episodes,
            "full_info": info,
        }
        all_results["seeds"].append(seed_result)  # type: ignore

        # Track for aggregation
        all_success_rates.append(success_rate)
        all_avg_rewards_successful.append(avg_reward_successful)
        all_wall_clock_times.append(time_per_episode)

        # Print seed results
        logging.info(
            colored(f"\nResults for seed {current_seed}:", "green", attrs=["bold"])
        )
        logging.info(f"  Success Rate: {success_rate:.4f}")
        logging.info(f"  Avg Sum Rewards (Successful): {avg_reward_successful:.4f}")
        logging.info(f"  Wall-clock Time per Episode: {time_per_episode:.4f} seconds")
        logging.info(f"  Total Time: {total_time:.2f} seconds")

    # Compute aggregated metrics across all seeds
    all_results["aggregated_metrics"] = {
        "success_rate": {
            "mean": float(np.mean(all_success_rates)),
            "std": float(np.std(all_success_rates)),
            "min": float(np.min(all_success_rates)),
            "max": float(np.max(all_success_rates)),
            "values": [float(x) for x in all_success_rates],
        },
        "avg_sum_rewards_successful": {
            "mean": float(np.mean(all_avg_rewards_successful)),
            "std": float(np.std(all_avg_rewards_successful)),
            "min": float(np.min(all_avg_rewards_successful)),
            "max": float(np.max(all_avg_rewards_successful)),
            "values": [float(x) for x in all_avg_rewards_successful],
        },
        "wall_clock_time_per_episode": {
            "mean": float(np.mean(all_wall_clock_times)),
            "std": float(np.std(all_wall_clock_times)),
            "min": float(np.min(all_wall_clock_times)),
            "max": float(np.max(all_wall_clock_times)),
            "values": [float(x) for x in all_wall_clock_times],
        },
    }

    # Print final aggregated results
    logging.info(
        colored(
            f"\n{'='*80}\nAggregated Results Across {cfg.num_seeds} Seeds\n{'='*80}",
            "cyan",
            attrs=["bold"],
        )
    )
    logging.info(colored("\nSuccess Rate:", "green", attrs=["bold"]))
    aggregated_metrics = all_results["aggregated_metrics"]
    success_rate_metrics = aggregated_metrics["success_rate"]  # type: ignore
    logging.info(f"  Mean: {success_rate_metrics['mean']:.4f}")
    logging.info(f"  Std:  {success_rate_metrics['std']:.4f}")
    logging.info(f"  Min:  {success_rate_metrics['min']:.4f}")
    logging.info(f"  Max:  {success_rate_metrics['max']:.4f}")

    logging.info(
        colored("\nAvg Sum Rewards (Successful Episodes):", "green", attrs=["bold"])
    )
    avg_sum_rewards_successful_metrics = aggregated_metrics[  # type: ignore
        "avg_sum_rewards_successful"
    ]
    logging.info(f"  Mean: {avg_sum_rewards_successful_metrics['mean']:.4f}")
    logging.info(f"  Std:  {avg_sum_rewards_successful_metrics['std']:.4f}")
    logging.info(f"  Min:  {avg_sum_rewards_successful_metrics['min']:.4f}")
    logging.info(f"  Max:  {avg_sum_rewards_successful_metrics['max']:.4f}")

    logging.info(
        colored("\nWall-clock Time per Episode (seconds):", "green", attrs=["bold"])
    )
    wall_clock_time_per_episode_metrics = aggregated_metrics[  # type: ignore
        "wall_clock_time_per_episode"
    ]
    logging.info(f"  Mean: {wall_clock_time_per_episode_metrics['mean']:.4f}")
    logging.info(f"  Std:  {wall_clock_time_per_episode_metrics['std']:.4f}")
    logging.info(f"  Min:  {wall_clock_time_per_episode_metrics['min']:.4f}")
    logging.info(f"  Max:  {wall_clock_time_per_episode_metrics['max']:.4f}")

    # Save results to JSON
    results_file = output_dir / "multi_seed_eval_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    logging.info(
        colored(f"\nResults saved to: {results_file}", "yellow", attrs=["bold"])
    )
    logging.info("End of multi-seed evaluation")


def main() -> None:
    """Main function."""
    init_logging()
    eval_multi_seed_main()  # pylint: disable=no-value-for-parameter


if __name__ == "__main__":
    main()
