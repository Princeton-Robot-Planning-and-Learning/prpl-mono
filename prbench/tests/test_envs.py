"""Common tests for all environments."""

import gymnasium
from gymnasium.utils.env_checker import check_env

import prbench


def test_env_make_and_check_env():
    """Tests that all registered environments can be created with make.

    Also calls gymnasium.utils.env_checker.check_env() to test API functions.
    """
    prbench.register_all_environments()
    env_ids = prbench.get_all_env_ids()
    assert len(env_ids) > 0
    for env_id in env_ids:
        # TidyBot mujoco_env is currently unstable, so we skip it.
        if "TidyBot" in env_id:
            continue
        # We currently require all environments to have RGB rendering.
        env = prbench.make(env_id, render_mode="rgb_array")
        assert env.render_mode == "rgb_array"
        assert isinstance(env, gymnasium.Env)
        check_env(env.unwrapped)
