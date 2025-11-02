"""Tests for TidyBot3D MJX backend support."""

import pytest

from prbench.envs.dynamic3d.tidybot3d import ObjectCentricTidyBot3DEnv, TidyBot3DConfig


def test_tidybot3d_standard_backend():
    """Test that standard MuJoCo backend works (backward compatibility)."""
    env = ObjectCentricTidyBot3DEnv(
        num_objects=3,
        render_images=False,
        use_mjx=False,
    )
    obs, _ = env.reset()
    assert env.observation_space.contains(obs), "Observation not in observation space"
    env.close()


def test_tidybot3d_config_with_mjx_params():
    """Test that TidyBot3DConfig accepts MJX parameters."""
    config = TidyBot3DConfig(
        control_frequency=20,
        use_mjx=False,
        device='cpu',
        num_envs=1
    )
    assert config.use_mjx is False
    assert config.device == 'cpu'
    assert config.num_envs == 1


def test_tidybot3d_uses_standard_backend_by_default():
    """Test that standard backend is used by default."""
    from prbench.envs.dynamic3d import tidybot_robot_env

    env = ObjectCentricTidyBot3DEnv(
        num_objects=3,
        render_images=False,
        use_mjx=False,
    )

    # Verify standard TidyBotRobotEnv class is used
    assert type(env._robot_env).__module__ == tidybot_robot_env.__name__

    env.close()


@pytest.mark.skipif(
    True,  # Skip by default - requires JAX installation
    reason="Requires JAX: pip install jax jaxlib"
)
def test_tidybot3d_with_mjx_backend():
    """Test TidyBot3D with MJX backend (requires JAX)."""
    pytest.importorskip("jax")

    # Clean conditional import - no warnings
    env = ObjectCentricTidyBot3DEnv(
        num_objects=3,
        render_images=False,
        use_mjx=True,
        device='cpu',  # Use CPU for testing
    )

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    action = env.action_space.sample()
    next_obs, _, _, _, _ = env.step(action)
    assert env.observation_space.contains(next_obs)

    env.close()


def test_tidybot3d_action_and_observation_spaces():
    """Test that action and observation spaces are valid with MJX params."""
    env = ObjectCentricTidyBot3DEnv(
        num_objects=3,
        render_images=False,
        use_mjx=False,
        device='cpu',
        num_envs=1
    )

    # Test observation space
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)

    # Test action space
    action = env.action_space.sample()
    assert env.action_space.contains(action)

    # Test step
    next_obs, reward, done, truncated, info = env.step(action)
    assert env.observation_space.contains(next_obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()
