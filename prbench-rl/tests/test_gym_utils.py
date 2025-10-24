"""Tests for gym utilities."""

import prbench

from prbench_rl.gym_utils import make_env


def test_make_env_truncation():
    """Test that make_env properly truncates PRBench environments with
    max_episode_steps."""

    # Register PRBench environments
    prbench.register_all_environments()

    # Create environment with max_episode_steps
    max_steps = 50
    env_fn = make_env(
        env_id="prbench/Obstruction2D-o0-v0",
        idx=0,
        capture_video=False,
        run_name="test_truncation",
        max_episode_steps=max_steps,
    )
    env = env_fn()

    # Reset environment
    _, _ = env.reset(seed=123)

    # Run for more than max_episode_steps and verify truncation happens
    step_count = 0
    truncated = False

    for _ in range(max_steps + 10):  # Run beyond max_steps
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        step_count += 1

        if terminated or truncated:
            break

    # Verify that the environment was truncated at max_steps
    assert (
        step_count == max_steps
    ), f"Environment should truncate at {max_steps} steps, but ran for {step_count}"
    assert truncated, "Environment should be truncated, not terminated"

    env.close()
