"""Tests for random_actions.py."""

import gymnasium

from programmatic_policy_learning.approaches.random_actions import RandomActionsApproach


def test_random_actions_approach():
    """Tests for RandomActionsApproach()."""
    # Just test that this runs without crashing.
    env = gymnasium.make("LunarLander-v3")
    approach = RandomActionsApproach(
        "N/A", env.observation_space, env.action_space, seed=123
    )
    obs, info = env.reset()
    approach.reset(obs, info)
    for _ in range(5):
        action = approach.step()
        obs, reward, terminated, _, info = env.step(action)
        approach.update(obs, reward, terminated, info)
