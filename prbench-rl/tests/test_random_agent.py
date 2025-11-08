"""Tests for the random agent."""

import copy

import numpy as np
import prbench
from gymnasium import spaces
from omegaconf import DictConfig

from prbench_rl.random_agent import RandomAgent


def test_random_agent_with_prbench_environment():
    """Test RandomAgent with continuous action space from PRBench environment."""
    prbench.register_all_environments()
    env = prbench.make("prbench/StickButton2D-b5-v0")

    # Ensure we have continuous action space
    assert isinstance(env.action_space, spaces.Box)
    assert isinstance(env.observation_space, spaces.Box)

    agent = RandomAgent(
        observation_space=env.observation_space,
        action_space=env.action_space,
        seed=456,
        cfg=DictConfig({}),
    )

    obs, info = env.reset(seed=456)
    agent.reset(obs, info)

    # Test agent interaction with environment
    for _ in range(5):
        assert env.observation_space.contains(obs)

        action = agent.step()
        assert env.action_space.contains(action)
        assert isinstance(action, np.ndarray)

        obs, reward, terminated, truncated, info = env.step(action)

        # Test transition learning (should not raise errors)
        agent.update(
            obs=obs,
            reward=reward,
            done=terminated or truncated,
            info=info,
        )

        if terminated or truncated:
            obs, info = env.reset()
            agent.reset(obs, info)

    env.close()


def test_random_agent_continuous_action_bounds():
    """Test RandomAgent respects continuous action space bounds."""
    obs_space = spaces.Box(low=-10.0, high=10.0, shape=(6,))
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    agent = RandomAgent(
        seed=123,
        observation_space=obs_space,
        action_space=action_space,
        cfg=DictConfig({}),
    )
    obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    agent.reset(obs, {})

    # Test multiple actions are within bounds
    for _ in range(20):
        action = agent.step()
        assert action_space.contains(action)
        assert isinstance(action, np.ndarray)
        assert action.shape == (3,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)


def test_random_agent_asymmetric_bounds():
    """Test RandomAgent with asymmetric action space bounds."""
    obs_space = spaces.Box(low=-1.0, high=1.0, shape=(3,))

    # Asymmetric bounds
    action_space = spaces.Box(
        low=np.array([-2.0, -1.0, 0.0]), high=np.array([3.0, 1.0, 5.0])
    )

    agent = RandomAgent(
        seed=111,
        observation_space=obs_space,
        action_space=action_space,
        cfg=DictConfig({}),
    )
    obs = np.array([0.0, 0.0, 0.0])
    agent.reset(obs, {})

    for _ in range(20):
        action = agent.step()
        assert action_space.contains(action)
        assert -2.0 <= action[0] <= 3.0
        assert -1.0 <= action[1] <= 1.0
        assert 0.0 <= action[2] <= 5.0


def test_random_agent_seeded_reproducibility_with_prbench():
    """Test seeded reproducibility with PRBench environment actions."""
    prbench.register_all_environments()
    env = prbench.make("prbench/StickButton2D-b5-v0")

    # Create separate action space instances to avoid shared RNG state
    action_space1 = copy.deepcopy(env.action_space)
    action_space2 = copy.deepcopy(env.action_space)

    # Create two agents with same seed but separate action spaces
    agent1 = RandomAgent(
        seed=789,
        observation_space=env.observation_space,
        action_space=action_space1,
        cfg=DictConfig({}),
    )
    agent2 = RandomAgent(
        seed=789,
        observation_space=env.observation_space,
        action_space=action_space2,
        cfg=DictConfig({}),
    )

    obs, info = env.reset(seed=789)
    agent1.reset(obs, info)
    agent2.reset(obs, info)

    # Actions should be identical with same seed
    for _ in range(10):
        action1 = agent1.step()
        action2 = agent2.step()
        np.testing.assert_array_equal(action1, action2)

    env.close()
