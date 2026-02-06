"""Script to create a pandas DataFrame from experiment results.

Examples:
  python experiments/create_results_dataframe.py --results_dir logs/2025-08-01/14-36-09

  python experiments/create_results_dataframe.py \
    --config_columns env.env_name env.make_kwargs.id seed max_abstract_plans \
        samples_per_step \
    --results_dir logs/2025-08-01/14-36-09
"""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf


def _main(
    results_dir: Path,
    columns: list[str],
    config_columns: list[str],
    output_file: Path | None = None,
) -> None:

    assert not set(columns) & set(config_columns)  # danger!

    # Load the configs and results from the subdirectories.
    results_with_configs = []
    for subdir in results_dir.iterdir():
        if not subdir.is_dir():
            continue
        results_file = subdir / "results.csv"
        config_file = subdir / "config.yaml"
        if not results_file.exists() or not config_file.exists():
            continue
        results = pd.read_csv(results_file)
        config = OmegaConf.load(config_file)
        assert isinstance(config, DictConfig)
        results_with_configs.append((results, config))

    # Combine everything into one dataframe.
    combined_data: list[dict[str, Any]] = []
    for results, config in results_with_configs:
        # Get the config columns once.
        config_data: dict[str, Any] = {}
        for col in config_columns:
            config_data[col] = OmegaConf.select(config, col)
        for _, row in results.iterrows():
            combined_row: dict[str, Any] = config_data.copy()
            # Extract regular columns.
            for col in columns:
                combined_row[col] = row[col]
            combined_data.append(combined_row)

    # Combine into a larger dataframe.
    combined_df = pd.DataFrame(combined_data)

    # Aggregate.
    # I hate pandas, why isn't this easy...
    group_cols = sorted(set(config_columns) - {"seed"})
    keep_cols = [c for c in combined_df.columns if c not in ["seed", "eval_episode"]]
    aggregated_df = combined_df.groupby(group_cols).mean().reset_index()[keep_cols]

    # Write output or print.
    if output_file is not None:
        aggregated_df.to_csv(output_file)
    else:
        print(aggregated_df.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create aggregated DataFrame from experiment results",
    )

    parser.add_argument(
        "--results_dir",
        type=Path,
        required=True,
        help="Directory containing experiment results",
    )

    parser.add_argument(
        "--columns",
        nargs="*",
        default=["success", "steps", "planning_time", "eval_episode"],
        help="Specific metric columns to include",
    )

    parser.add_argument(
        "--config_columns",
        nargs="*",
        default=["env.env_name", "seed", "max_abstract_plans", "samples_per_step"],
        help="Specific config columns to include (default: env).",
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
        args.output_file,
    )
