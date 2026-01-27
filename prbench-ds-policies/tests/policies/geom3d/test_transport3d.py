"""Tests for transport3d.py domain-specific policy."""

import numpy as np
import prbench
import pytest
from gymnasium.wrappers import RecordVideo

from prbench_ds_policies.policies import create_domain_specific_policy
from tests.conftest import MAKE_VIDEOS

prbench.register_all_environments()


def test_transport3d_policy_returns_valid_action():
    """Test that the policy returns a valid action."""
    env = prbench.make("prbench/Transport3D-o1-v0", use_gui=False)
    policy = create_domain_specific_policy(
        "transport3d",
        observation_space=env.observation_space,
        num_cubes=1,
        action_space=env.action_space,
        birrt_extend_num_interp=10,
        smooth_mp_max_time=1.0,
        smooth_mp_max_candidate_plans=1,
    )
    obs, _ = env.reset(seed=123)

    action = policy(obs)

    assert isinstance(action, np.ndarray)
    assert action.shape == (11,)
    assert action.dtype == np.float32
    assert env.action_space.contains(action)

    env.close()


def test_transport3d_dynamic_loader():
    """Test that the policy can be loaded via the dynamic loader."""
    env = prbench.make("prbench/Transport3D-o1-v0", use_gui=False)
    policy = create_domain_specific_policy(
        "transport3d",
        observation_space=env.observation_space,
        action_space=env.action_space,
        num_cubes=1,
        birrt_extend_num_interp=10,
        smooth_mp_max_time=1.0,
        smooth_mp_max_candidate_plans=1,
    )
    obs, _ = env.reset(seed=123)

    action = policy(obs)

    assert isinstance(action, np.ndarray)
    assert action.shape == (11,)
    assert env.action_space.contains(action)

    env.close()


@pytest.mark.parametrize("seed", [123])
def test_transport3d_o1_policy_solves_task(seed):
    """Test that the policy can solve the Transport3D-o1 task."""

    env = prbench.make(
        "prbench/Transport3D-o1-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Transport3D-o1-ds-policy-{seed}"
        )

    policy = create_domain_specific_policy(
        "transport3d",
        observation_space=env.observation_space,
        num_cubes=1,
        action_space=env.action_space,
        seed=seed,
        birrt_extend_num_interp=10,
        smooth_mp_max_time=1.0,
        smooth_mp_max_candidate_plans=1,
    )
    obs, _ = env.reset(seed=seed)

    for _ in range(3000):
        action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        env.close()
        pytest.fail("Did not terminate within 3000 steps")

    assert terminated, "Task should have terminated successfully"

    env.close()


@pytest.mark.parametrize("seed", [124])
def test_transport3d_o2_policy_solves_task(seed):
    """Test that the policy can solve the Transport3D-o2 task."""

    env = prbench.make(
        "prbench/Transport3D-o2-v0",
        render_mode="rgb_array",
        use_gui=False,
        realistic_bg=True,
    )

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"Transport3D-o2-ds-policy-{seed}"
        )

    policy = create_domain_specific_policy(
        "transport3d",
        observation_space=env.observation_space,
        num_cubes=2,
        action_space=env.action_space,
        seed=seed,
        birrt_extend_num_interp=10,
        smooth_mp_max_time=1.0,
        smooth_mp_max_candidate_plans=1,
    )
    obs, _ = env.reset(seed=seed)

    for _ in range(3000):
        action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        env.close()
        pytest.fail("Did not terminate within 3000 steps")

    assert terminated, "Task should have terminated successfully"

    env.close()
