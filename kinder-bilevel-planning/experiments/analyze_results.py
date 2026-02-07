#!/usr/bin/env python3
"""Script to analyze experimental results from multi-run Hydra experiments.

Usage:
    python analyze_results.py <log_dir>

Example:
    python experiments/analyze_results.py logs/2026-01-29/20-45-20
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
        "env_name": config["env"]["make_kwargs"]["id"],
        "seed": config["seed"],
        "max_abstract_plans": config.get("max_abstract_plans", "unknown"),
        "samples_per_step": config.get("samples_per_step", "unknown"),
        "max_skill_horizon": config.get("max_skill_horizon", "unknown"),
        "heuristic_name": config.get("heuristic_name", "unknown"),
        "planning_timeout": config.get("planning_timeout", "unknown"),
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
            row_data = {
                "env_name": data["env_name"],
                "max_abstract_plans": data["max_abstract_plans"],
                "samples_per_step": data["samples_per_step"],
                "max_skill_horizon": data["max_skill_horizon"],
                "heuristic_name": data["heuristic_name"],
                "planning_timeout": data["planning_timeout"],
                "seed": data["seed"],
                "eval_episode": row["eval_episode"],
                "success": row["success"],
                "planning_time": row["planning_time"],
                "steps": row["steps"],
            }
            # Add reward if it exists
            if "reward" in row:
                row_data["reward"] = row["reward"]
            all_data.append(row_data)

    if not all_data:
        print("No data loaded from runs")
        return pd.DataFrame()

    # Create DataFrame
    df = pd.DataFrame(all_data)

    # Determine grouping columns based on what varies across runs
    group_cols = ["env_name"]
    for col in [
        "max_abstract_plans",
        "samples_per_step",
        "max_skill_horizon",
        "heuristic_name",
        "planning_timeout",
    ]:
        if df[col].nunique() > 1:
            group_cols.append(col)

    # Prepare aggregation dictionary
    agg_dict = {
        "success": ["mean", "std", "count"],
        "planning_time": ["mean", "std"],
        "steps": "mean",
    }

    # Add reward aggregation if it exists
    has_reward = "reward" in df.columns
    if has_reward:
        agg_dict["reward"] = ["mean", "std"]

    # Group and aggregate
    grouped = df.groupby(group_cols).agg(agg_dict)

    # Flatten column names before reset_index
    col_names = [
        "solve_rate",
        "solve_rate_std",
        "num_runs",
        "avg_planning_time",
        "planning_time_std",
        "avg_steps",
    ]
    if has_reward:
        col_names.extend(["avg_reward", "avg_reward_std"])

    grouped.columns = col_names
    grouped = grouped.reset_index()

    # Calculate average reward among successful episodes if reward exists
    if has_reward:
        successful_reward = (
            df[df["success"] == True]  # pylint: disable=singleton-comparison
            .groupby(group_cols)["reward"]
            .agg(["mean", "std"])
            .reset_index()
        )
        successful_reward.columns = list(group_cols) + [
            "avg_reward_successful",
            "reward_successful_std",
        ]

        # Merge successful reward back into grouped dataframe
        grouped = grouped.merge(successful_reward, on=group_cols, how="left")

    # Sort by grouping columns
    grouped = grouped.sort_values(group_cols)

    return grouped


def format_table(df: pd.DataFrame) -> str:
    """Format the results as a nice table with mean ± std for key metrics."""
    if df.empty:
        return "No results to display"

    has_reward = "avg_reward" in df.columns

    # Dynamically determine environment column width based on longest environment name
    max_env_len = max(len(str(env)) for env in df["env_name"])
    env_col_width = max(
        max_env_len + 2, len("Environment")
    )  # At least as wide as header

    # Create a formatted table
    lines = []

    # Build header dynamically based on varying columns
    header_parts = ["Environment"]
    config_cols = []
    for col in [
        "max_abstract_plans",
        "samples_per_step",
        "max_skill_horizon",
        "heuristic_name",
        "planning_timeout",
    ]:
        if col in df.columns:
            config_cols.append(col)
            # Shorten column names for display
            display_name = {
                "max_abstract_plans": "Max Plans",
                "samples_per_step": "Samples",
                "max_skill_horizon": "Skill Horizon",
                "heuristic_name": "Heuristic",
                "planning_timeout": "Timeout",
            }.get(col, col)
            header_parts.append(display_name)

    header_parts.extend(
        [
            "Solve Rate (mean±std)",
            "Planning Time/s (mean±std)",
            "Avg Steps",
        ]
    )

    if has_reward:
        header_parts.extend(["Avg Reward (mean±std)", "Avg Reward Success (mean±std)"])

    # Calculate column widths
    widths = [env_col_width] + [15] * len(config_cols) + [25, 30, 15]
    if has_reward:
        widths.extend([25, 30])

    # Calculate total table width
    header_line_length = sum(widths) + len(widths)  # +len(widths) for spaces

    lines.append("=" * header_line_length)

    # Format header
    header = ""
    for part, width in zip(header_parts, widths):
        header += f"{part:<{width}} "
    lines.append(header.rstrip())
    lines.append("=" * header_line_length)

    for _, row in df.iterrows():
        parts = [row["env_name"]]

        # Add config columns
        for col in config_cols:
            parts.append(str(row[col]))

        # Format metrics with mean ± std
        parts.append(f"{row['solve_rate']:.1%} ± {row['solve_rate_std']:.1%}")
        parts.append(f"{row['avg_planning_time']:.4f} ± {row['planning_time_std']:.4f}")
        parts.append(f"{row['avg_steps']:.1f}")

        if has_reward:
            parts.append(f"{row['avg_reward']:.3f} ± {row['avg_reward_std']:.3f}")
            avg_reward_successful = (
                f"{row['avg_reward_successful']:.3f} ± {row['reward_successful_std']:.3f}"  # pylint: disable=line-too-long
                if pd.notna(row["avg_reward_successful"])
                else "N/A"
            )
            parts.append(avg_reward_successful)

        # Format row
        row_str = ""
        for part, width in zip(parts, widths):
            row_str += f"{part:<{width}} "
        lines.append(row_str.rstrip())

    lines.append("=" * header_line_length)
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
