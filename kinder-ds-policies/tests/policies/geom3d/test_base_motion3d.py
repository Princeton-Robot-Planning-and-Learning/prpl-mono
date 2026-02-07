"""Tests for base_motion3d.py domain-specific policy."""

import kinder
import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from kinder.envs.kinematic3d.base_motion3d import BaseMotion3DObjectCentricState

from kinder_ds_policies.policies import create_domain_specific_policy
from tests.conftest import MAKE_VIDEOS

kinder.register_all_environments()


def test_base_motion3d_policy_returns_valid_action():
    """Test that the policy returns a valid action."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    policy = create_domain_specific_policy(
        "base_motion3d", observation_space=env.observation_space
    )
    obs, _ = env.reset(seed=123)

    action = policy(obs)

    assert isinstance(action, np.ndarray)
    assert action.shape == (11,)
    assert action.dtype == np.float32
    assert env.action_space.contains(action)

    env.close()


def test_base_motion3d_policy_moves_toward_target():
    """Test that the policy action moves the robot toward the target."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    policy = create_domain_specific_policy(
        "base_motion3d", observation_space=env.observation_space
    )
    obs, _ = env.reset(seed=123)

    # Get initial distance to target
    oc_obs = env.observation_space.devectorize(obs)
    state = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    initial_dist = np.linalg.norm(
        [
            state.target_base_pose.x - state.base_pose.x,
            state.target_base_pose.y - state.base_pose.y,
        ]
    )

    # Take a few steps
    for _ in range(10):
        action = policy(obs)
        obs, _, _, _, _ = env.step(action)

    # Get final distance to target
    oc_obs = env.observation_space.devectorize(obs)
    state = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
    final_dist = np.linalg.norm(
        [
            state.target_base_pose.x - state.base_pose.x,
            state.target_base_pose.y - state.base_pose.y,
        ]
    )

    # Distance should decrease after taking steps toward target
    assert final_dist < initial_dist

    env.close()


def test_base_motion3d_dynamic_loader():
    """Test that the policy can be loaded via the dynamic loader."""
    env = kinder.make("kinder/BaseMotion3D-v0")
    policy = create_domain_specific_policy(
        "base_motion3d", observation_space=env.observation_space
    )
    obs, _ = env.reset(seed=123)

    action = policy(obs)

    assert isinstance(action, np.ndarray)
    assert action.shape == (11,)
    assert env.action_space.contains(action)

    env.close()


@pytest.mark.parametrize("seed", [123])
def test_base_motion3d_policy_solves_task(seed):
    """Test that the policy can solve the BaseMotion3D task."""

    env = kinder.make("kinder/BaseMotion3D-v0", render_mode="rgb_array")

    if MAKE_VIDEOS:
        env = RecordVideo(
            env, "unit_test_videos", name_prefix=f"BaseMotion3D-ds-policy-{seed}"
        )

    policy = create_domain_specific_policy(
        "base_motion3d", observation_space=env.observation_space
    )
    obs, _ = env.reset(seed=seed)

    for _ in range(1000):
        action = policy(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    else:
        env.close()
        pytest.fail("Did not terminate within 1000 steps")

    assert terminated, "Task should have terminated successfully"

    env.close()
