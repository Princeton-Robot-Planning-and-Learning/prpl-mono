"""Utility functions for PRBench."""

from pathlib import Path
from typing import List

import dill as pkl


def load_demo(demo_path: Path) -> dict:
    """Load a demonstration from a pickle file."""
    try:
        with open(demo_path, "rb") as f:
            demo_data = pkl.load(f)

        # Validate demo data structure.
        required_keys = ["env_id", "observations", "actions"]
        for key in required_keys:
            if key not in demo_data:
                raise ValueError(f"Demo data missing required key: {key}")

        if not demo_data["actions"]:
            raise ValueError("Demo contains no actions")

        if len(demo_data["observations"]) != len(demo_data["actions"]) + 1:
            print(
                f"Warning: Expected {len(demo_data['actions']) + 1} observations, "
                f"got {len(demo_data['observations'])}"
            )

        if "seed" not in demo_data:
            raise ValueError(" Demo does not contain seed information.")

        return demo_data
    except Exception as e:
        # Don't exit, just raise the exception to be handled by caller
        raise ValueError(f"Error loading demo from {demo_path}: {e}") from e


def find_all_demo_files() -> List[Path]:
    """Find all demo files in the demos directory."""
    demos_dir = Path(__file__).parent.parent.parent / "demos"
    demo_files = list(demos_dir.glob("**/*.p"))
    return sorted(demo_files)


def get_env_id_from_demo_path(demo_path: Path) -> str:
    """Extract environment ID from demo file path structure."""
    # Demo path structure: demos/{env_name}/{instance}/{timestamp}.p
    env_name = demo_path.parent.parent.name
    # Convert from demo directory name to environment ID
    env_id = f"prbench/{env_name}-v0"
    return env_id
