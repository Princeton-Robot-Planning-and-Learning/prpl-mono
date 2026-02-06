"""Script to create a pandas DataFrame from experiment results.

Examples:
  # Aggregate evaluation results from RL experiments
  python experiments/create_results_dataframe.py --results_dir outputs/2026-01-25

  # Aggregate training results
  python experiments/create_results_dataframe.py \
    --results_dir outputs/2026-01-25 \
    --results_type train

  # Custom config columns
  python experiments/create_results_dataframe.py \
    --results_dir outputs/2026-01-25 \
    --config_columns agent.name env_id seed agent.args.total_timesteps

  # Save to CSV file
  python experiments/create_results_dataframe.py \
    --results_dir outputs/2026-01-25 \
    --output_file aggregated_results.csv
"""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf


def _find_experiment_dirs(results_dir: Path) -> list[Path]:
    """Recursively find all experiment directories containing config.yaml."""
    experiment_dirs = []
    for path in results_dir.rglob("config.yaml"):
        experiment_dirs.append(path.parent)
    return experiment_dirs


def _main(
    results_dir: Path,
    columns: list[str],
    config_columns: list[str],
    results_type: str = "eval",
    output_file: Path | None = None,
) -> None:

    assert not set(columns) & set(config_columns)  # danger!

    # Determine which results file to look for
    results_filename = f"{results_type}_results.csv"

    # Find all experiment directories recursively
    experiment_dirs = _find_experiment_dirs(results_dir)

    if not experiment_dirs:
        print(f"No experiment directories found in {results_dir}")
        return

    # Load the configs and results from the subdirectories.
    results_with_configs = []
    for subdir in experiment_dirs:
        results_file = subdir / results_filename
        config_file = subdir / "config.yaml"
        if not results_file.exists() or not config_file.exists():
            continue
        results = pd.read_csv(results_file)
        config = OmegaConf.load(config_file)
        assert isinstance(config, DictConfig)
        results_with_configs.append((results, config, subdir))

    if not results_with_configs:
        print(f"No {results_filename} files found in experiment directories")
        return

    print(f"Found {len(results_with_configs)} experiments with {results_filename}")

    # Combine everything into one dataframe.
    combined_data: list[dict[str, Any]] = []
    for results, config, subdir in results_with_configs:
        # Get the config columns once.
        config_data: dict[str, Any] = {}
        for col in config_columns:
            val = OmegaConf.select(config, col)
            config_data[col] = val
        config_data["experiment_dir"] = str(subdir)

        for _, row in results.iterrows():
            combined_row: dict[str, Any] = config_data.copy()
            # Extract regular columns (use all columns if none specified).
            cols_to_use = columns if columns else list(row.index)
            for col in cols_to_use:
                if col in row.index:
                    combined_row[col] = row[col]
            combined_data.append(combined_row)

    # Combine into a larger dataframe.
    combined_df = pd.DataFrame(combined_data)

    # Aggregate by config columns (excluding seed).
    group_cols = sorted(set(config_columns) - {"seed"})
    if group_cols:
        # Calculate aggregated statistics
        numeric_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [c for c in numeric_cols if c not in group_cols]

        agg_dict = {}
        for col in numeric_cols:
            agg_dict[col] = ["mean", "std", "min", "max", "count"]

        aggregated_df = (
            combined_df.groupby(group_cols)
            .agg(agg_dict)  # type: ignore[arg-type]
            .reset_index()
        )
        # Flatten column names
        aggregated_df.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] for col in aggregated_df.columns
        ]
    else:
        aggregated_df = combined_df

    # Print summary
    print("\n" + "=" * 60)
    print("AGGREGATED RESULTS SUMMARY")
    print("=" * 60)
    print(f"Results type: {results_type}")
    print(f"Total experiments: {len(results_with_configs)}")
    print(f"Total data points: {len(combined_df)}")
    print("-" * 60)

    # Write output.
    if output_file is not None:
        # Save both raw and aggregated results
        raw_output = output_file.with_stem(f"{output_file.stem}_raw")
        combined_df.to_csv(raw_output, index=False)
        print(f"Saved raw results to: {raw_output}")

        aggregated_df.to_csv(output_file, index=False)
        print(f"Saved aggregated results to: {output_file}")
    else:
        print("\nAggregated Results:")
        print(aggregated_df.to_string())

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create aggregated DataFrame from RL experiment results",
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing experiment results (e.g., outputs/2026-01-25)",
    )

    parser.add_argument(
        "--results_type",
        type=str,
        default="eval",
        choices=["eval", "train"],
        help="Type of results to aggregate: 'eval' or 'train' (default: eval)",
    )

    parser.add_argument(
        "--columns",
        nargs="*",
        default=["episodic_return", "step_length"],
        help="Metric columns to include (default: episodic_return, step_length)",
    )

    parser.add_argument(
        "--config_columns",
        nargs="*",
        default=["agent.name", "env_id", "seed", "max_episode_steps"],
        help="Config columns for grouping (default: agent.name, env_id, seed)",
    )

    parser.add_argument(
        "--output_file",
        type=Path,
        help="Output CSV file path (if not specified, prints to stdout)",
    )

    args = parser.parse_args()
    _main(
        args.results_dir,
        args.columns,
        args.config_columns,
        args.results_type,
        args.output_file,
    )
