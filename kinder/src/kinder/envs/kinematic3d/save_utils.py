"""Utility functions for kinematic3d tests."""

import time
from pathlib import Path
from typing import Any

import dill as pkl  # type: ignore[import-untyped]

# Default demos directory: kinder/demos relative to this file
# Utils file: prpl-mono/kinder/tests/envs/kinematic3d/utils.py
# Demos:      prpl-mono/kinder/demos
_UTILS_DIR = Path(__file__).resolve().parent
DEFAULT_DEMOS_DIR = _UTILS_DIR.parent.parent.parent.parent / "demos"


def sanitize_env_id(env_id: str) -> str:
    """Remove unnecessary stuff from the env ID.

    Mirrors the function in kinder/scripts/generate_env_docs.py and collect_demos_ds.py
    for consistent directory naming.
    """
    if env_id.startswith("kinder/"):
        env_id = env_id[len("kinder/") :]
    env_id = env_id.replace("/", "_")
    if len(env_id) >= 3 and env_id[-3:-1] == "-v":
        return env_id[:-3]
    return env_id


def save_demo(
    demo_dir: Path,
    env_id: str,
    seed: int,
    observations: list[Any],
    actions: list[Any],
    rewards: list[float],
    terminated: bool,
    truncated: bool,
) -> Path:
    """Save a demo to disk in the same format as collect_demos_ds.py.

    Directory structure: {demo_dir}/{sanitized_env_id}/{seed}/{timestamp}.p
    """
    timestamp = int(time.time())
    demo_subdir = demo_dir / sanitize_env_id(env_id) / str(seed)
    demo_subdir.mkdir(parents=True, exist_ok=True)
    demo_path = demo_subdir / f"{timestamp}.p"
    demo_data = {
        "env_id": env_id,
        "timestamp": timestamp,
        "seed": seed,
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminated": terminated,
        "truncated": truncated,
    }
    with open(demo_path, "wb") as f:
        pkl.dump(demo_data, f)
    return demo_path
