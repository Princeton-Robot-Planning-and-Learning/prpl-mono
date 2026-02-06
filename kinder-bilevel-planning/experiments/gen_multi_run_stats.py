"""Script to calculate mean and std statistics across multiple seed runs.

This script reads results.csv files from seed_* subdirectories within a base folder,
calculates mean and standard deviation for each metric across ALL episodes in ALL seeds,
and saves the aggregated statistics as a CSV file.

For the 'success' metric, statistics are calculated over all episodes.
For all other metrics, statistics are calculated only over successful episodes
(where success=True), to measure performance when the task succeeds.

Examples:
    python experiments/gen_multi_run_stats.py --base_dir logs/motion2d-p0
    python experiments/gen_multi_run_stats.py --base_dir logs/obstruction2d-p0 \
        --metrics success steps planning_time
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def calculate_multi_run_stats(
    base_dir: Path,
    metrics: list[str] | None = None,
    output_file: Path | None = None,
) -> pd.DataFrame:
    """Calculate mean and std statistics across multiple seed runs.

    Args:
        base_dir: Base directory containing seed_* subdirectories
        metrics: List of metric columns to aggregate (if None, uses all numeric columns)
        output_file: Path to save statistics CSV
            (if None, saves to base_dir/results_summary.csv)

    Returns:
        DataFrame with mean and std for each metric
    """
    # Find all seed_* directories
    seed_dirs = sorted(base_dir.glob("seed_*"))
    if not seed_dirs:
        raise ValueError(f"No seed_* directories found in {base_dir}")

    print(f"Found {len(seed_dirs)} seed directories: {[d.name for d in seed_dirs]}")

    # Load all results.csv files
    all_results = []
    for seed_dir in seed_dirs:
        results_file = seed_dir / "results.csv"
        if not results_file.exists():
            print(f"Warning: {results_file} not found, skipping...")
            continue

        df = pd.read_csv(results_file)
        all_results.append(df)

    if not all_results:
        raise ValueError(f"No valid results.csv files found in {base_dir}/seed_*")

    print(f"Loaded {len(all_results)} results files")

    # Determine which columns to aggregate
    if metrics is None:
        # Use all numeric columns except eval_episode (which is an index)
        sample_df = all_results[0]
        metrics = [
            col
            for col in sample_df.columns
            if col != "eval_episode" and pd.api.types.is_numeric_dtype(sample_df[col])
        ]
        # Handle success as numeric (True=1, False=0)
        if "success" in sample_df.columns:
            if "success" not in metrics:
                metrics.insert(0, "success")

    print(f"Aggregating metrics: {metrics}")

    # Convert success column to numeric if it exists
    for df in all_results:
        if "success" in df.columns and df["success"].dtype == "object":
            df["success"] = df["success"].map(
                {"True": 1, True: 1, "False": 0, False: 0}
            )

    # Check if success column exists
    has_success = "success" in all_results[0].columns

    # Calculate statistics across all episodes in all seeds
    results: dict[str, list] = {"metric": [], "mean": [], "std": []}

    for metric in metrics:
        if metric == "success" or not has_success:
            # For success metric (or if no success column), calculate over ALL episodes
            all_values = np.concatenate([df[metric].values for df in all_results])
            mean_val = np.mean(all_values)
            std_val = np.std(all_values, ddof=1)  # Sample std
        else:
            # For other metrics, only calculate over successful episodes
            successful_values: list = []
            for df in all_results:
                # Get values where success is True
                mask = df["success"].astype(bool)
                successful_values.extend(df.loc[mask, metric].values)

            if len(successful_values) > 0:
                mean_val = np.mean(successful_values)
                std_val = (
                    np.std(successful_values, ddof=1)
                    if len(successful_values) > 1
                    else 0.0
                )
            else:
                mean_val = np.nan
                std_val = np.nan

        results["metric"].append(metric)
        results["mean"].append(mean_val)
        results["std"].append(std_val)

    # Create DataFrame
    summary_df = pd.DataFrame(results)

    # Set default output path if not specified
    if output_file is None:
        output_file = base_dir / "results_summary.csv"

    # Save to CSV
    summary_df.to_csv(output_file, index=False)

    print(f"\nSaved summary statistics to: {output_file}")

    # Print summary
    print("\n=== Summary Statistics ===")
    for _, row in summary_df.iterrows():
        metric = row["metric"]
        mean_val = row["mean"]
        std_val = row["std"]
        if metric == "success" and has_success:
            print(f"{metric}: {mean_val:.2%} ± {std_val:.4f}")
        else:
            print(f"{metric}: {mean_val:.4f} ± {std_val:.4f}")

    if has_success:
        success_rate = summary_df[summary_df["metric"] == "success"]["mean"].values[0]
        total_episodes = sum(len(df) for df in all_results)
        successful_episodes = int(success_rate * total_episodes)
        print(f"\n({successful_episodes}/{total_episodes} successful episodes)")

    return summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate mean and std statistics across multiple seed runs"
    )

    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="Base directory containing seed_* subdirectories (e.g., logs/motion2d-p0)",
    )

    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Specific metrics to aggregate (default: all numeric columns)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for summary statistics CSV "
        "(default: <base_dir>/results_summary.csv)",
    )

    args = parser.parse_args()

    calculate_multi_run_stats(
        base_dir=args.base_dir,
        metrics=args.metrics,
        output_file=args.output,
    )
