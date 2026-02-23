"""Tests for motion3d.py."""

import numpy as np
import pytest
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from kinder.envs.kinematic3d.motion3d import (
    Motion3DEnv,
    Motion3DObjectCentricState,
    ObjectCentricMotion3DEnv,
)
from tests.conftest import MAKE_VIDEOS


@pytest.fixture(scope="module")
def env():
    """Create a shared environment for all tests in this module."""
    environment = Motion3DEnv(
        render_mode="rgb_array", use_gui=False, realistic_bg=False
    )
    if MAKE_VIDEOS:
        environment = RecordVideo(environment, "unit_test_videos")
    yield environment
    environment.close()


def test_motion3d_env(env):  # pylint: disable=redefined-outer-name
    """Tests for basic methods in motion3D env."""
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.unwrapped._object_centric_env.physics_client_id)


def test_motion_planning_in_motion3d_env(env):  # pylint: disable=redefined-outer-name
    """Proof of concept that motion planning works in this environment."""
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = (
        env.unwrapped._object_centric_env.config  # pylint: disable=protected-access
    )

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Motion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricMotion3DEnv(config=config, realistic_bg=False)

    # Run motion planning.
    if MAKE_VIDEOS:  # make a smooth motion plan for videos
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    # Uncomment to debug target pose.
    # from pybullet_helpers.gui import visualize_pose
    # import pybullet as p
    # visualize_pose(Pose(obs.target_position, (1, 0, 0, 0)), sim.physics_client_id)
    # while True:
    #     p.getMouseEvents(sim.physics_client_id)

    joint_plan = run_smooth_motion_planning_to_pose(
        Pose(obs.target_position, (1, 0, 0, 0)),
        sim.robot.arm,
        collision_ids=set(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    assert joint_plan is not None
    # Make sure we stay below the required max_action_mag by a fair amount.
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot.arm, max_distance=config.max_action_mag / 2
    )

    env.action_space.seed(123)
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = [0.0] * 3 + delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, done, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Motion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        if done:
            break
    else:
        assert False, "Plan did not reach goal"
