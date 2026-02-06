"""Organize results with both training and evaluation results."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def success_average_reward(rewards: list[float]) -> float:
    """Calculate the average reward of successful episodes."""
    success_rewards: list[float] = []
    for reward in rewards:
        if reward != -500:
            success_rewards.append(reward)
    return float(np.mean(success_rewards))


def main() -> None:
    """Organize results with both training and evaluation results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_results_dir", type=Path, required=True)
    parser.add_argument("--eval_results_dir", type=Path, required=True)
    args = parser.parse_args()

    with open(  # pylint: disable=unspecified-encoding
        args.train_results_dir / "info.json", "r"
    ) as f:
        train_data = json.load(f)
    with open(  # pylint: disable=unspecified-encoding
        args.eval_results_dir / "multi_seed_eval_results.json", "r"
    ) as f:
        eval_data = json.load(f)

    new_train_data = {
        "demonstrations": train_data["total_episodes"],
        "environment steps": train_data["total_frames"],
    }

    seed_data_detailed = {}
    average_rewards_list = []
    for seed_data in eval_data["seeds"]:
        seed_data_detailed[seed_data["seed"]] = {
            "Successes": seed_data["full_info"]["per_task"][0]["metrics"]["successes"],
            "Steps": [
                -step
                for step in seed_data["full_info"]["per_task"][0]["metrics"][
                    "sum_rewards"
                ]
            ],
        }
        rewards = seed_data["full_info"]["per_task"][0]["metrics"]["sum_rewards"]
        average_rewards_list.append(-success_average_reward(rewards))

    new_eval_data = {
        "summary statistics": {
            "success rate": eval_data["aggregated_metrics"]["success_rate"],
            "wall clock time per episode": eval_data["aggregated_metrics"][
                "wall_clock_time_per_episode"
            ],
            "steps per success episode": {
                "mean": np.mean(average_rewards_list),
                "std": np.std(average_rewards_list),
                "min": np.min(average_rewards_list),
                "max": np.max(average_rewards_list),
                "values": average_rewards_list,
            },
        }
    }
    all_results = {
        "train_results": new_train_data,
        "eval_results": new_eval_data,
        "detailed eval results for each seed": seed_data_detailed,
    }

    task_name = eval_data["config"]["env"]["task"]
    if not os.path.exists(f"outputs/alpha_imitation_learning/{task_name}"):
        os.makedirs(f"outputs/alpha_imitation_learning/{task_name}")
    with open(  # pylint: disable=unspecified-encoding
        f"outputs/alpha_imitation_learning/{task_name}/organized_results.json", "w"
    ) as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
