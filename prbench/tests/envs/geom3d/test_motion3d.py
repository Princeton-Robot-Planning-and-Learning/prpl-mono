"""Tests for motion3d.py."""

import numpy as np
from conftest import MAKE_VIDEOS
from gymnasium.wrappers import RecordVideo
from prpl_utils.utils import wrap_angle
from pybullet_helpers.geometry import Pose
from pybullet_helpers.motion_planning import (
    remap_joint_position_plan_to_constant_distance,
    run_smooth_motion_planning_to_pose,
)
from relational_structs.spaces import ObjectCentricBoxSpace

from prbench.envs.geom3d.motion3d import (
    Motion3DEnv,
    Motion3DObjectCentricState,
    ObjectCentricMotion3DEnv,
)


def test_motion3d_env():
    """Tests for basic methods in motion3D env."""

    env = Motion3DEnv(use_gui=False)  # set use_gui=True to debug
    obs, _ = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)

    for _ in range(10):
        act = env.action_space.sample()
        assert isinstance(act, np.ndarray)
        obs, _, _, _, _ = env.step(act)

    # Uncomment to debug.
    # import pybullet as p
    # while True:
    #     p.getMouseEvents(env.physics_client_id)


def test_motion_planning_in_motion3d_env():
    """Proof of concept that motion planning works in this environment."""

    # Create the real environment.
    env = Motion3DEnv(render_mode="rgb_array")
    assert isinstance(env.observation_space, ObjectCentricBoxSpace)
    config = env._object_centric_env.config  # pylint: disable=protected-access
    if MAKE_VIDEOS:
        env = RecordVideo(env, "unit_test_videos")

    vec_obs, _ = env.reset(seed=123)
    # NOTE: we should soon make this smoother.
    oc_obs = env.observation_space.devectorize(vec_obs)
    obs = Motion3DObjectCentricState(oc_obs.data, oc_obs.type_features)

    # Create a simulator for planning.
    sim = ObjectCentricMotion3DEnv(config=config)

    # Run motion planning.
    if MAKE_VIDEOS:  # make a smooth motion plan for videos
        max_candidate_plans = 20
    else:
        max_candidate_plans = 1

    joint_plan = run_smooth_motion_planning_to_pose(
        Pose(obs.target_position),
        sim.robot,
        collision_ids=set(),
        end_effector_frame_to_plan_frame=Pose.identity(),
        seed=123,
        max_candidate_plans=max_candidate_plans,
    )
    assert joint_plan is not None
    # Make sure we stay below the required max_action_mag by a fair amount.
    joint_plan = remap_joint_position_plan_to_constant_distance(
        joint_plan, sim.robot, max_distance=config.max_action_mag / 2
    )

    env.action_space.seed(123)
    for target_joints in joint_plan[1:]:
        delta = np.subtract(target_joints[:7], obs.joint_positions)
        delta_lst = [wrap_angle(a) for a in delta]
        action_lst = delta_lst + [0.0]
        action = np.array(action_lst, dtype=np.float32)
        vec_obs, _, done, _, _ = env.step(action)
        # NOTE: we should soon make this smoother.
        oc_obs = env.observation_space.devectorize(vec_obs)
        obs = Motion3DObjectCentricState(oc_obs.data, oc_obs.type_features)
        if done:
            break
    else:
        assert False, "Plan did not reach goal"
    env.close()
