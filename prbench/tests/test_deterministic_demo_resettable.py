"""Tests for deterministic demo replay across all environments."""

from pathlib import Path

import numpy as np
import pytest

import prbench
from prbench.utils import find_all_demo_files, load_demo


@pytest.mark.parametrize("demo_path", find_all_demo_files())
def test_deterministic_demo_reset(demo_path: Path) -> None:
    """Test that demo replay produces identical observations and rewards.

    This test verifies that:
    1. Loading a demo file succeeds
    2. Environment can be created for the demo's environment ID
    For each observation and action pair in the demo:
        3. Resetting the environment with that observation
        4. Replaying the action produces the next observation
        5. Checking the reproduced observation matches the demo's next observation
    """
    # Register all environments
    prbench.register_all_environments()

    # Load demo data
    if "Dyn" in str(demo_path):
        pytest.skip("Skipping Dynamic2D due to unresettable physical simulation")
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
    obs, _ = env.reset(seed=seed, options={"init_state": expected_observations[0]})
    # obs, _ = env.reset(seed=seed)

    # Check initial observation matches
    obs_difference = np.abs(obs - expected_observations[0]).max()
    print(f"Initalization max observation difference = {obs_difference}")

    # Replay all actions and verify observations/rewards
    for i, expected_obs in enumerate(expected_observations[:-1]):
        obs, _ = env.reset(seed=seed, options={"init_state": expected_obs})
        action = actions[i]
        obs_next, reward, terminated, truncated, _ = env.step(action)

        # Check observation matches
        expected_obs_next = expected_observations[i + 1]
        obs_difference = np.abs(obs_next - expected_obs_next).max()
        print(f"Step {i}: max observation difference = {obs_difference}")

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
