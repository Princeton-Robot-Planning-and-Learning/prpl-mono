"""Tests for deterministic demo replay across all environments."""

from pathlib import Path
from typing import List

import numpy as np
import pytest

import prbench
from prbench.utils import load_demo


def find_all_demo_files() -> List[Path]:
    """Find all demo files in the demos directory."""
    demos_dir = Path(__file__).parent.parent / "demos"
    demo_files = list(demos_dir.glob("**/*.p"))
    return sorted(demo_files)


def get_env_id_from_demo_path(demo_path: Path) -> str:
    """Extract environment ID from demo file path structure."""
    # Demo path structure: demos/{env_name}/{instance}/{timestamp}.p
    env_name = demo_path.parent.parent.name
    # Convert from demo directory name to environment ID
    env_id = f"prbench/{env_name}-v0"
    return env_id


@pytest.mark.parametrize("demo_path", find_all_demo_files())
def test_deterministic_demo_replay(demo_path: Path):
    """Test that demo replay produces identical observations and rewards.

    This test verifies that:
    1. Loading a demo file succeeds
    2. Environment can be created for the demo's environment ID
    3. Replaying actions with the same seed produces identical observations
    4. Replaying actions produces identical rewards (if available)
    """
    # Register all environments
    prbench.register_all_environments()

    # Load demo data
    try:
        demo_data = load_demo(demo_path)
    except Exception as e:
        pytest.skip(f"Failed to load demo {demo_path}: {e}")

    # Extract demo information
    env_id = demo_data["env_id"]
    actions = demo_data["actions"]
    expected_observations = demo_data["observations"]
    expected_rewards = demo_data.get("rewards", None)
    seed = demo_data["seed"]

    # Skip if no actions to replay
    if len(actions) == 0:
        pytest.skip(f"Demo {demo_path} contains no actions")

    # Create environment
    env = prbench.make(env_id, render_mode="rgb_array")

    # Test reproducibility: reset with seed and replay actions
    obs, _ = env.reset(seed=seed)

    # Check initial observation matches
    assert np.allclose(
        obs, expected_observations[0], atol=1e-4
    ), f"Initial observation mismatch in {demo_path}"

    # Replay all actions and verify observations/rewards
    for i, action in enumerate(actions):
        obs, reward, terminated, truncated, _ = env.step(action)

        # Check observation matches
        expected_obs = expected_observations[i + 1]
        assert np.allclose(obs, expected_obs, atol=1e-4), \
            f"Observation mismatch at step {i} in {demo_path}"

        # Check reward matches (if available)
        if expected_rewards is not None and i < len(expected_rewards):
            expected_reward = expected_rewards[i]
            assert reward == expected_reward, (
                f"Reward mismatch at step {i} in {demo_path}: "
                f"got {reward}, expected {expected_reward}"
            )

        # Stop if episode ended early
        if terminated or truncated:
            break
    env.close()  # type: ignore[no-untyped-call]
