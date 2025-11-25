"""Script to generate multi-run statistics across seeds.

This script reads eval_results.csv from all seed folders, classifies episodes as
successful or failed, and calculates mean and std of success rate and steps.

Examples:
  python experiments/gen_multi_run_stats.py \
    --exp_dir prbench-rl/outputs/2025-11-23/ppo_m2d_0_passage \
    --output_file stats.csv
"""

import argparse
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd


def parse_episodic_return(return_str: str) -> float:
    """Parse episodic return from string format like '[-39.]' to float -39.0."""
    # Remove brackets and convert to float
    return_str = return_str.strip()
    if return_str.startswith("[") and return_str.endswith("]"):
        return_str = return_str[1:-1]
    return float(return_str.rstrip("."))


def calculate_stats(exp_dir: Path, success_threshold: float = -300.0) -> dict:
    """Calculate statistics across all seeds.

    Args:
        exp_dir: Directory containing seed_* subdirectories
        success_threshold: Episodes with return > threshold are considered successful

    Returns:
        Dictionary containing mean and std statistics
    """
    all_returns: list[float] = []
    seed_success_rates: list[float] = []
    seed_successful_steps: list[float] = []

    # Find all seed directories
    seed_dirs = sorted(
        [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
    )

    if not seed_dirs:
        raise ValueError(f"No seed directories found in {exp_dir}")

    print(f"Found {len(seed_dirs)} seed directories: {[d.name for d in seed_dirs]}")

    for seed_dir in seed_dirs:
        eval_file = seed_dir / "eval_results.csv"
        if not eval_file.exists():
            print(f"Warning: {eval_file} not found, skipping...")
            continue

        # Read the eval results
        df = pd.read_csv(eval_file)

        # Parse episodic returns
        returns = cast(
            npt.NDArray[np.floating[Any]],
            df["episodic_return"].apply(parse_episodic_return).to_numpy(),
        )
        all_returns.extend(returns.tolist())

        # Classify episodes as successful or failed
        successful: npt.NDArray[np.bool_] = returns > success_threshold
        success_rate: float = float(successful.mean())
        seed_success_rates.append(success_rate)

        # For successful episodes, calculate steps (which is -return)
        if successful.any():
            successful_steps: npt.NDArray[np.floating[Any]] = -returns[successful]
            seed_successful_steps.append(float(successful_steps.mean()))
        else:
            # If no successful episodes, record NaN
            seed_successful_steps.append(np.nan)

        avg_steps_str = (
            f"{seed_successful_steps[-1]:.2f}"
            if not np.isnan(seed_successful_steps[-1])
            else "N/A"
        )
        print(
            f"  {seed_dir.name}: {len(returns)} episodes, "
            f"success rate = {success_rate:.2%}, "
            f"avg steps (successful) = {avg_steps_str}"
        )

    # Calculate overall statistics
    all_returns_array: npt.NDArray[np.floating[Any]] = np.array(all_returns)
    successful_mask: npt.NDArray[np.bool_] = all_returns_array > success_threshold

    stats = {
        "total_episodes": len(all_returns),
        "num_seeds": len(seed_success_rates),
        "success_rate_mean": np.mean(seed_success_rates),
        "success_rate_std": (
            np.std(seed_success_rates, ddof=1) if len(seed_success_rates) > 1 else 0.0
        ),
    }

    # Calculate steps statistics for successful episodes
    if successful_mask.any():
        # Calculate mean steps across all successful episodes from all seeds
        all_successful_steps: npt.NDArray[np.floating[Any]] = -all_returns_array[
            successful_mask
        ]
        stats["successful_steps_mean"] = float(np.mean(all_successful_steps))
        stats["successful_steps_std"] = float(np.std(all_successful_steps, ddof=1))

        # Also calculate mean and std of the per-seed successful steps means
        valid_seed_steps = [s for s in seed_successful_steps if not np.isnan(s)]
        if valid_seed_steps:
            stats["successful_steps_mean_per_seed"] = np.mean(valid_seed_steps)
            stats["successful_steps_std_per_seed"] = (
                np.std(valid_seed_steps, ddof=1) if len(valid_seed_steps) > 1 else 0.0
            )
    else:
        stats["successful_steps_mean"] = np.nan
        stats["successful_steps_std"] = np.nan
        stats["successful_steps_mean_per_seed"] = np.nan
        stats["successful_steps_std_per_seed"] = np.nan

    return stats


def main(
    exp_dir: Path, output_file: Path | None = None, success_threshold: float = -300.0
) -> None:
    """Generate multi-run statistics and save to CSV.

    Args:
        exp_dir: Directory containing seed_* subdirectories
        output_file: Output CSV file path (if None, prints to stdout)
        success_threshold: Episodes with return > threshold are considered successful
    """
    print(f"\nCalculating statistics for: {exp_dir}")
    print(f"Success threshold: return > {success_threshold}")
    print("-" * 80)

    stats = calculate_stats(exp_dir, success_threshold)

    print("\n" + "=" * 80)
    print("OVERALL STATISTICS")
    print("=" * 80)
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Number of seeds: {stats['num_seeds']}")
    print(
        f"Success rate: {stats['success_rate_mean']:.2%} ± "
        f"{stats['success_rate_std']:.2%}"
    )
    print(
        f"Steps (successful, all episodes): "
        f"{stats['successful_steps_mean']:.2f} ± "
        f"{stats['successful_steps_std']:.2f}"
    )
    print(
        f"Steps (successful, per-seed means): "
        f"{stats['successful_steps_mean_per_seed']:.2f} ± "
        f"{stats['successful_steps_std_per_seed']:.2f}"
    )

    # Create DataFrame and save
    df = pd.DataFrame([stats])

    if output_file is not None:
        df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\n" + "=" * 80)
        print("CSV OUTPUT")
        print("=" * 80)
        print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate statistics across multiple seed runs",
    )

    parser.add_argument(
        "--exp_dir",
        type=Path,
        required=True,
        help="Directory containing seed_* subdirectories with eval_results.csv files",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        help="Output CSV file path (if not specified, prints to stdout)",
    )

    parser.add_argument(
        "--success_threshold",
        type=float,
        default=-300.0,
        help=(
            "Episodes with return > threshold are considered "
            "successful (default: -300.0)"
        ),
    )

    args = parser.parse_args()
    main(args.exp_dir, args.output_file, args.success_threshold)
