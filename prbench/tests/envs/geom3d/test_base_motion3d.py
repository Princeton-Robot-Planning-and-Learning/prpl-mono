"""Tests for base_motion3d.py."""

from unittest.mock import patch

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from pybullet_helpers.motion_planning import (
    run_single_arm_mobile_base_motion_planning,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.base_motion3d import (
    BaseMotion3DEnv,
    BaseMotion3DObjectCentricState,
    ObjectCentricBaseMotion3DEnv,
)


def test_base_motion3d_env():
    """Tests for basic methods in base motion3D env."""

    env = BaseMotion3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env._object_centric_env.physics_client_id)


def test_motion_planning_in_base_motion3d_env():
    """Proof of concept that motion planning works in this environment."""

    # Create the real environment.
    env = BaseMotion3DEnv(render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env._object_centric_env.config  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricBaseMotion3DEnv(config=config)

    base_plan = run_single_arm_mobile_base_motion_planning(
        sim.robot,
        sim.robot.base.get_pose(),
        obs.target_base_pose,
        collision_bodies=set(),
        seed=123,
    )
    assert base_plan is not None

    env.action_space.seed(123)
    for target_base_pose in base_plan[1:]:
        current_base_pose = obs.base_pose
        delta = target_base_pose - current_base_pose
        delta_lst = [delta.x, delta.y, delta.rot]
        action_lst = delta_lst + [0.0] * 7 + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, done, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = BaseMotion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        if done:
            break
    else:
        assert False, "Plan did not reach goal"
    env.close()


def test_check_mobile_base_collisions_is_called():
    """Test that check_mobile_base_collisions is called when there is a collision."""
    env = BaseMotion3DEnv(use_gui=False)
    env.reset(seed=123)

    # Patch the check_mobile_base_collisions function
    with patch(
        "prbench.envs.geom3d.base_env.check_mobile_base_collisions"
    ) as mock_check_base:
        # Set return value to False (no collision)
        mock_check_base.return_value = False

        # Take an action that moves the base (first 3 elements are base actions)
        action = np.array([0.01, 0.01, 0.0] + [0.0] * 7 + [0.0], dtype=np.float32)
        env.step(action)

        # Verify that check_mobile_base_collisions was called
        assert mock_check_base.called, "check_mobile_base_collisions should be called"

        # Verify it was called with the correct arguments
        assert mock_check_base.call_count >= 1
        call_args = mock_check_base.call_args
        assert call_args is not None

        # Verify the robot base was passed as the first argument
        assert (
            call_args[0][0]
            == env._object_centric_env.robot.base  # pylint: disable=protected-access
        )

    env.close()
