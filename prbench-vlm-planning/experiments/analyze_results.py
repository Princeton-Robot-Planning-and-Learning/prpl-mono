#!/usr/bin/env python3
"""Script to analyze experimental results from multi-run Hydra experiments.

Usage:
    python analyze_results.py <log_dir>

Example:
    python analyze_results.py ../logs/2025-11-20/20-21-32
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml  # type: ignore[import-untyped]


def load_run_data(run_dir: Path) -> Optional[Dict]:
    """Load configuration and results from a single run directory."""
    config_path = run_dir / "config.yaml"
    results_path = run_dir / "results.csv"

    if not config_path.exists() or not results_path.exists():
        return None

    # Load config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Load results
    results_df = pd.read_csv(results_path)

    return {
        "env": config["env"],
        "seed": config["seed"],
        "rgb_observation": config["rgb_observation"],
        "vlm_model": config.get("vlm_model", "unknown"),
        "temperature": config.get("temperature", "unknown"),
        "results": results_df,
    }


def analyze_results(log_dir: Path) -> pd.DataFrame:
    """Analyze all runs in the log directory and compute aggregated metrics."""
    all_data = []

    # Iterate through all numbered subdirectories
    run_dirs = [d for d in log_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    run_dirs.sort(key=lambda x: int(x.name))

    if not run_dirs:
        print(f"No run directories found in {log_dir}")
        return pd.DataFrame()

    print(f"Found {len(run_dirs)} run directories")

    for run_dir in run_dirs:
        data = load_run_data(run_dir)
        if data is None:
            print(f"Warning: Skipping {run_dir.name} - missing files")
            continue

        # Extract metrics from results
        results = data["results"]
        for _, row in results.iterrows():
            all_data.append(
                {
                    "env": data["env"],
                    "rgb_observation": data["rgb_observation"],
                    "seed": data["seed"],
                    "eval_episode": row["eval_episode"],
                    "success": row["success"],
                    "planning_time": row["planning_time"],
                    "steps": row["steps"],
                }
            )

    if not all_data:
        print("No data loaded from runs")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Group by env and rgb_observation, compute averages
    grouped = (
        df.groupby(["env", "rgb_observation"])
        .agg(
            {
                "success": ["mean", "std", "count"],
                "planning_time": ["mean", "std"],
                "steps": "mean",
            }
        )
        .reset_index()
    )

    # Flatten column names
    grouped.columns = pd.Index(
        [
            "env",
            "rgb_observation",
            "solve_rate",
            "solve_rate_std",
            "num_runs",
            "avg_planning_time",
            "planning_time_std",
            "avg_steps",
        ]
    )

    # Sort by environment and rgb_observation
    grouped = grouped.sort_values(["env", "rgb_observation"], ascending=[True, False])

    return grouped


def format_table(df: pd.DataFrame) -> str:
    """Format the results as a nice table with mean ± std for key metrics."""
    if df.empty:
        return "No results to display"

    # Create a formatted table
    lines = []
    lines.append("=" * 130)
    lines.append(
        f"{'Environment':<25} {'Method':<15} "
        f"{'Solve Rate (mean±std)':<25} "
        f"{'Planning Time/s (mean±std)':<30} {'Avg Steps':<15}"
    )
    lines.append("=" * 130)

    for _, row in df.iterrows():
        env = row["env"]
        method = "With Image" if row["rgb_observation"] else "Without Image"
        # Format with mean ± std explicitly
        solve_rate = f"{row['solve_rate']:.1%} ± {row['solve_rate_std']:.1%}"
        planning_time = (
            f"{row['avg_planning_time']:.4f} ± {row['planning_time_std']:.4f}"
        )
        avg_steps = f"{row['avg_steps']:.1f}"

        lines.append(
            f"{env:<25} {method:<15} {solve_rate:<25} "
            f"{planning_time:<30} {avg_steps:<15}"
        )

    lines.append("=" * 130)
    lines.append(f"\nTotal configurations: {len(df)}")
    lines.append(f"Runs per configuration: {int(df['num_runs'].iloc[0])}")

    return "\n".join(lines)


def main() -> None:
    """Main function to analyze experimental results."""
    parser = argparse.ArgumentParser(
        description="Analyze experimental results from Hydra multi-run experiments"
    )
    parser.add_argument(
        "log_dir", type=str, help="Path to the log directory containing run results"
    )
    parser.add_argument(
        "--csv", type=str, default=None, help="Optional: Save results to CSV file"
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        sys.exit(1)

    print(f"Analyzing results from: {log_dir}")
    print()

    # Analyze results
    results_df = analyze_results(log_dir)

    if results_df.empty:
        print("No results found")
        sys.exit(1)

    # Print formatted table
    print(format_table(results_df))

    # Optionally save to CSV
    if args.csv:
        csv_path = Path(args.csv)
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")


if __name__ == "__main__":
    main()  # type: ignore[no-untyped-call]
